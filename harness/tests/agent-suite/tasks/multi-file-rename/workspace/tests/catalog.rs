use shop::catalog::Catalog;
use shop::{price, Widget, WidgetId};

#[test]
fn add_and_total() {
    let mut c = Catalog::default();
    assert!(c.add_widget(Widget::new(1, "bolt", 405)));
    assert!(c.add_widget(Widget::new(2, "nut", 95)));
    assert!(!c.add_widget(Widget::new(1, "copy", 1)));
    assert_eq!(c.widget_total(), 500);
    assert_eq!(c.get(WidgetId(2)).map(|w| w.name.as_str()), Some("nut"));
}

#[test]
fn price_label() {
    assert_eq!(price::label(&Widget::new(3, "gear", 1205)), "$12.05");
}
