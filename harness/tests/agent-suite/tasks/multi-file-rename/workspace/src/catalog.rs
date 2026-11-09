//! Stores Widget items by id.
use crate::{Widget, WidgetId};
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct Catalog {
    items: BTreeMap<WidgetId, Widget>,
}

impl Catalog {
    /// Adds a Widget. Returns false when its id is already present.
    pub fn add_widget(&mut self, widget: Widget) -> bool {
        if self.items.contains_key(&widget.id) {
            return false;
        }
        self.items.insert(widget.id, widget);
        true
    }

    pub fn get(&self, id: WidgetId) -> Option<&Widget> {
        self.items.get(&id)
    }

    /// The sum of all Widget prices, in cents.
    pub fn widget_total(&self) -> u64 {
        self.items.values().map(crate::price::cents_of).sum()
    }
}
