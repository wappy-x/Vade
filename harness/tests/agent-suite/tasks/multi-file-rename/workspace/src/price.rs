//! Prices of Widget items.
use crate::Widget;

pub fn cents_of(widget: &Widget) -> u64 {
    widget.cents
}

/// Formats the price of a Widget as dollars, for example `$4.05`.
pub fn label(widget: &Widget) -> String {
    format!("${}.{:02}", widget.cents / 100, widget.cents % 100)
}
