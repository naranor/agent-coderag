use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
}

impl User {
    pub fn new(name: String) -> Self {
        User { id: 0, name }
    }
}
