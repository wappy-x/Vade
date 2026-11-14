//! Unit conversions for the weather station.

/// Degrees Celsius to degrees Fahrenheit.
pub fn celsius_to_fahrenheit(c: f64) -> f64 {
    c * 9.0 / 5.0 + 32.0
}

/// Degrees Fahrenheit to degrees Celsius.
pub fn fahrenheit_to_celsius(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0
}

/// Degrees Celsius to kelvin.
pub fn celsius_to_kelvin(c: f64) -> f64 {
    c + 273.15
}

/// Kilometers per hour to meters per second.
pub fn kmh_to_ms(kmh: f64) -> f64 {
    kmh / 3.6
}
