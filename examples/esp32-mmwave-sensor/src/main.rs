//! `ruvector-mmwave-sensor` — iter A bring-up firmware.
//!
//! Reads the Seeed MR60BHA2 60 GHz mmWave radar over UART1 on the
//! Waveshare ESP32-S3-Touch-AMOLED-1.8 board (ADR-SYS-0026), parses the
//! Seeed binary protocol, and logs decoded vital signs to the
//! USB-Serial-JTAG console (`/dev/ttyACM0` on the host).
//!
//! Default UART pinout (override at compile time via env vars):
//!
//! ```text
//!   GPIO 17  →  UART1 RX  (radar TX → MCU RX)
//!   GPIO 18  →  UART1 TX  (MCU TX → radar RX, unused in iter A)
//! ```
//!
//! Build & flash:
//!
//! ```sh
//!   . ~/export-esp.sh
//!   cd examples/esp32-mmwave-sensor
//!   cargo +esp build --release
//!   espflash flash --monitor /dev/ttyACM0 \
//!       target/xtensa-esp32s3-espidf/release/ruvector-mmwave-sensor
//! ```
//!
//! Iter B will add an mTLS embed-RPC client that posts decoded events
//! to the `ruvector-hailo` cluster — placeholder hooks are flagged
//! with `TODO(iter-B)` comments below.

mod parser;

use anyhow::Result;
use esp_idf_hal::peripherals::Peripherals;
use esp_idf_hal::uart::{config::Config as UartConfig, UartDriver};
use esp_idf_hal::units::Hertz;
use log::{info, warn};
use parser::{Event, Mr60Parser};
use std::time::{Duration, Instant};

/// Latest snapshot of decoded radar state, updated every parsed event.
/// Keeping it on the main task's stack (~80 bytes) lets the iter-A
/// bring-up stay single-threaded; iter B will move this behind a
/// `Mutex<RadarState>` for the embed-RPC poster task.
#[derive(Debug, Default, Clone, Copy)]
struct RadarState {
    heart_rate_bpm: Option<u8>,
    breathing_bpm: Option<u8>,
    distance_cm: Option<u16>,
    presence: Option<bool>,
    /// Parsed-frame counter — useful for "is the radar alive?" checks.
    frames_total: u32,
    /// Frames whose checksum failed — surfaced separately so a noisy
    /// cable shows up clearly in logs without being mistaken for "no
    /// person detected".
    frames_corrupt: u32,
    /// Frames whose `frame_type` we don't decode — non-fatal but
    /// indicates a firmware-revision mismatch with the radar.
    frames_unknown: u32,
}

impl RadarState {
    fn apply(&mut self, ev: Event) {
        match ev {
            Event::Breathing { bpm } => {
                self.breathing_bpm = Some(bpm);
                self.frames_total = self.frames_total.wrapping_add(1);
            }
            Event::HeartRate { bpm } => {
                self.heart_rate_bpm = Some(bpm);
                self.frames_total = self.frames_total.wrapping_add(1);
            }
            Event::Distance { cm } => {
                self.distance_cm = Some(cm);
                self.frames_total = self.frames_total.wrapping_add(1);
            }
            Event::Presence { present } => {
                self.presence = Some(present);
                self.frames_total = self.frames_total.wrapping_add(1);
            }
            Event::Unknown { .. } => {
                self.frames_unknown = self.frames_unknown.wrapping_add(1);
            }
            Event::ChecksumError => {
                self.frames_corrupt = self.frames_corrupt.wrapping_add(1);
            }
            // Resync is a normal startup transient — don't pollute counters.
            Event::Resync => {}
        }
    }
}

/// GPIO pins for the radar UART. The Waveshare AMOLED-1.8 reserves
/// GPIO 4-7, 11, 12 for QSPI (SH8601 display) and GPIO 14, 15 for I2C
/// (FT3168 touch + TCA9554 IO expander). 17/18 are free per
/// ADR-SYS-0026's pin map; pick those by default. Swap via env vars
/// at compile time if your wiring differs.
const DEFAULT_RX_GPIO: u8 = 17;
const DEFAULT_TX_GPIO: u8 = 18;

/// MR60BHA2 default UART baud (per Seeed datasheet).
const RADAR_BAUD: u32 = 115_200;

/// How often to print the latest snapshot, regardless of frame arrival.
/// 1 Hz keeps the log readable; faster overwhelms USB-Serial-JTAG.
const STATUS_INTERVAL: Duration = Duration::from_secs(1);

fn main() -> Result<()> {
    // esp-idf-svc patches up panic + logger + sys init.
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!("ruvector-mmwave-sensor iter-A — boot");
    info!(
        "expecting MR60BHA2 on UART1 rx=GPIO{} tx=GPIO{} @ {} baud",
        DEFAULT_RX_GPIO, DEFAULT_TX_GPIO, RADAR_BAUD
    );

    let peripherals = Peripherals::take()?;
    // We pin to UART1 because UART0 is reserved for the boot console
    // (even though our sdkconfig routes the actual stdout through the
    // USB-Serial-JTAG peripheral, UART0 itself is still claimed).
    let uart = peripherals.uart1;
    // GPIO acquisition: esp-idf-hal 0.45 lets us pluck pins by name.
    // We use `into_input_output` because iter B may write commands
    // back to the radar to enable advanced modes.
    let tx_pin = peripherals.pins.gpio18;
    let rx_pin = peripherals.pins.gpio17;

    let cfg = UartConfig::new().baudrate(Hertz(RADAR_BAUD));
    let driver = UartDriver::new(
        uart,
        tx_pin,
        rx_pin,
        Option::<esp_idf_hal::gpio::AnyIOPin>::None,
        Option::<esp_idf_hal::gpio::AnyIOPin>::None,
        &cfg,
    )?;
    info!("UART1 driver up");

    let mut parser = Mr60Parser::new();
    let mut state = RadarState::default();
    let mut last_print = Instant::now();
    let mut buf = [0u8; 256];

    loop {
        // Block up to 50 ms — short enough that the 1 Hz status print
        // stays near-real-time even when the radar is silent.
        match driver.read(&mut buf, 50) {
            Ok(n) if n > 0 => {
                parser.feed_slice(&buf[..n], |ev| state.apply(ev));
            }
            Ok(_) => {
                // Timeout / no bytes available — fall through to print.
            }
            Err(e) => {
                warn!("UART read error: {:?} — continuing", e);
            }
        }

        if last_print.elapsed() >= STATUS_INTERVAL {
            print_state(&state);
            last_print = Instant::now();
        }
        // TODO(iter-B): post the latest state to the ruvector-hailo
        // cluster's embed RPC over mTLS once a vitals frame has
        // changed. Use the rate_limit-on-cert path validated in
        // the iter-111 composition test.
    }
}

fn print_state(s: &RadarState) {
    info!(
        "vitals hr_bpm={:?} br_bpm={:?} dist_cm={:?} present={:?} frames_total={} corrupt={} unknown={}",
        s.heart_rate_bpm,
        s.breathing_bpm,
        s.distance_cm,
        s.presence,
        s.frames_total,
        s.frames_corrupt,
        s.frames_unknown,
    );
}
