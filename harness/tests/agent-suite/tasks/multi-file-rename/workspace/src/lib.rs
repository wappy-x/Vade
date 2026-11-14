//! A small catalog of Widget items.

pub mod catalog;
pub mod price;

/// The id of one Widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WidgetId(pub u32);

/// One item for sale.
#[derive(Debug, Clone, PartialEq)]
pub struct Widget {
    pub id: WidgetId,
    pub name: String,
    pub cents: u64,
}

impl Widget {
    pub fn new(id: u32, name: &str, cents: u64) -> Widget {
        Widget { id: WidgetId(id), name: name.to_string(), cents }
    }
}
