use units::temperature;

#[test]
fn module_paths() {
    assert_eq!(temperature::celsius_to_fahrenheit(-40.0), -40.0);
    assert_eq!(temperature::fahrenheit_to_celsius(32.0), 0.0);
    assert_eq!(temperature::celsius_to_kelvin(-273.15), 0.0);
}
