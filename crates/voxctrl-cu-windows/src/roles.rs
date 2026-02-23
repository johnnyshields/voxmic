//! Map UIA ControlType IDs to cross-platform `UiRole`.

use voxctrl_cu::tree::UiRole;

// UIA ControlType IDs (from the 50000 series).
// See: https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-controltype-ids
const UIA_BUTTON: i32 = 50000;
const UIA_CALENDAR: i32 = 50001;
const UIA_CHECKBOX: i32 = 50002;
const UIA_COMBOBOX: i32 = 50003;
const UIA_EDIT: i32 = 50004;
const UIA_HYPERLINK: i32 = 50005;
const UIA_IMAGE: i32 = 50006;
const UIA_LISTITEM: i32 = 50007;
const UIA_LIST: i32 = 50008;
const UIA_MENU: i32 = 50009;
const UIA_MENUBAR: i32 = 50010;
const UIA_MENUITEM: i32 = 50011;
const UIA_PROGRESSBAR: i32 = 50012;
const UIA_RADIOBUTTON: i32 = 50013;
const UIA_SCROLLBAR: i32 = 50014;
const UIA_SLIDER: i32 = 50015;
const UIA_SPINNER: i32 = 50016;
const UIA_STATUSBAR: i32 = 50017;
const UIA_TAB: i32 = 50018;
const UIA_TABITEM: i32 = 50019;
const UIA_TEXT: i32 = 50020;
const UIA_TOOLBAR: i32 = 50021;
const UIA_TOOLTIP: i32 = 50022;
const UIA_TREE: i32 = 50023;
const UIA_TREEITEM: i32 = 50024;
const UIA_CUSTOM: i32 = 50025;
const UIA_GROUP: i32 = 50026;
const UIA_THUMB: i32 = 50027;
const UIA_DATAGRID: i32 = 50028;
const UIA_DATAITEM: i32 = 50029;
const UIA_DOCUMENT: i32 = 50030;
const UIA_SPLITBUTTON: i32 = 50031;
const UIA_WINDOW: i32 = 50032;
const UIA_PANE: i32 = 50033;
const UIA_HEADER: i32 = 50034;
const UIA_HEADERITEM: i32 = 50035;
const UIA_TABLE: i32 = 50036;
const UIA_TITLEBAR: i32 = 50037;
const UIA_SEPARATOR: i32 = 50038;

/// Map a UIA `ControlType` integer to a `UiRole`.
pub fn control_type_to_role(ct: i32) -> UiRole {
    match ct {
        UIA_BUTTON | UIA_SPLITBUTTON => UiRole::Button,
        UIA_CHECKBOX => UiRole::CheckBox,
        UIA_COMBOBOX => UiRole::ComboBox,
        UIA_EDIT => UiRole::TextInput,
        UIA_HYPERLINK => UiRole::Link,
        UIA_IMAGE => UiRole::Image,
        UIA_LISTITEM | UIA_DATAITEM => UiRole::ListItem,
        UIA_LIST | UIA_DATAGRID => UiRole::List,
        UIA_MENU => UiRole::Menu,
        UIA_MENUBAR => UiRole::MenuBar,
        UIA_MENUITEM => UiRole::MenuItem,
        UIA_PROGRESSBAR => UiRole::ProgressBar,
        UIA_RADIOBUTTON => UiRole::RadioButton,
        UIA_SCROLLBAR => UiRole::ScrollBar,
        UIA_SLIDER | UIA_SPINNER => UiRole::Slider,
        UIA_STATUSBAR => UiRole::StatusBar,
        UIA_TAB => UiRole::Tab,
        UIA_TABITEM => UiRole::TabItem,
        UIA_TEXT => UiRole::Label,
        UIA_TOOLBAR => UiRole::ToolBar,
        UIA_TREE => UiRole::Tree,
        UIA_TREEITEM => UiRole::TreeItem,
        UIA_CUSTOM => UiRole::Custom("custom".into()),
        UIA_GROUP => UiRole::Group,
        UIA_DOCUMENT => UiRole::Document,
        UIA_WINDOW => UiRole::Window,
        UIA_PANE => UiRole::Pane,
        UIA_TABLE => UiRole::Table,
        UIA_HEADER => UiRole::Group,
        UIA_HEADERITEM => UiRole::TableCell,
        UIA_TITLEBAR => UiRole::TitleBar,
        UIA_SEPARATOR => UiRole::Separator,
        UIA_CALENDAR => UiRole::Custom("calendar".into()),
        UIA_TOOLTIP => UiRole::Custom("tooltip".into()),
        UIA_THUMB => UiRole::Custom("thumb".into()),
        _ => UiRole::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_control_types() {
        assert_eq!(control_type_to_role(50000), UiRole::Button);
        assert_eq!(control_type_to_role(50004), UiRole::TextInput);
        assert_eq!(control_type_to_role(50032), UiRole::Window);
    }

    #[test]
    fn unknown_control_type() {
        assert_eq!(control_type_to_role(99999), UiRole::Unknown);
    }
}
