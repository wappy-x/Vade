use units::{celsius_to_fahrenheit, celsius_to_kelvin, fahrenheit_to_celsius, kmh_to_ms};

#[test]
fn conversions() {
    assert_eq!(celsius_to_fahrenheit(100.0), 212.0);
    assert_eq!(fahrenheit_to_celsius(212.0), 100.0);
    assert_eq!(celsius_to_kelvin(0.0), 273.15);
    assert_eq!(kmh_to_ms(36.0), 10.0);
}
