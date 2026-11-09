//! The engine-1 set: every prompt, schema and tool of the engine card, built
//! in code. Long prompts come from a seeded sentence generator, so no large
//! file is checked in and the same seed always gives the same text. Nothing
//! here names a checkpoint, a tokenizer or a token id: prompt sizes in tokens
//! are measured from the server's own usage counts at run time.

/// The set version printed on every engine card and receipt.
pub const ENGINE_SET: &str = "engine-1";

const ADJECTIVES: [&str; 16] = [
    "quiet", "narrow", "golden", "distant", "patient", "silver", "humble", "steady", "crowded",
    "gentle", "hollow", "bright", "ancient", "careful", "rugged", "simple",
];
const NOUNS: [&str; 16] = [
    "river", "lantern", "garden", "harbor", "orchard", "bridge", "meadow", "library", "workshop",
    "village", "market", "valley", "kitchen", "tower", "forest", "station",
];
const VERBS: [&str; 12] = [
    "passes", "faces", "follows", "shelters", "overlooks", "joins", "circles", "borders",
    "crosses", "reaches", "guards", "touches",
];
const ENDINGS: [&str; 12] = [
    "at dawn", "in the evening", "after the rain", "before the market opens", "during the long winter",
    "when the wind turns", "in early spring", "under a pale sky", "as the bells ring", "on most days",
    "near the end of summer", "while the town sleeps",
];

/// SplitMix64, the one generator behind every filler sentence.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Filler sentence `index` of stream `seed`. Plain words only: no digits, no
/// codes and no question, so a needle is the only code in its document.
pub fn filler_sentence(seed: u64, index: u64) -> String {
    let r = mix(seed.wrapping_mul(0x1_0000_0001).wrapping_add(index));
    let pick = |shift: u32, len: usize| ((r >> shift) as usize) % len;
    format!(
        "The {} {} {} the {} {} {}.",
        ADJECTIVES[pick(0, ADJECTIVES.len())],
        NOUNS[pick(8, NOUNS.len())],
        VERBS[pick(16, VERBS.len())],
        ADJECTIVES[pick(24, ADJECTIVES.len())],
        NOUNS[pick(32, NOUNS.len())],
        ENDINGS[pick(40, ENDINGS.len())]
    )
}

/// `count` filler sentences of stream `seed`, eight to a paragraph.
pub fn filler(seed: u64, count: usize) -> String {
    let mut text = String::new();
    for index in 0..count {
        if index > 0 {
            text.push_str(if index % 8 == 0 { "\n\n" } else { " " });
        }
        text.push_str(&filler_sentence(seed, index as u64));
    }
    text
}

/// One planted fact and the depth it is planted at.
#[derive(Clone, Copy, Debug)]
pub struct Needle {
    pub label: &'static str,
    /// The prompt must hold at least this many tokens (server count).
    pub target_tokens: u64,
    pub seed: u64,
    /// The fact sentence planted at the middle of the filler.
    pub fact: &'static str,
    /// The exact text the answer must contain.
    pub answer: &'static str,
    pub question: &'static str,
}

pub const NEEDLES: [Needle; 2] = [
    Needle {
        label: "8K",
        target_tokens: 8_192,
        seed: 8_001,
        fact: "The archive code for the north reading room is AMBER-7319-HERON.",
        answer: "AMBER-7319-HERON",
        question: "What is the archive code for the north reading room? Reply with the code only.",
    },
    Needle {
        label: "32K",
        target_tokens: 32_768,
        seed: 32_001,
        fact: "The vault phrase for the east storage hall is COBALT-2846-WILLOW.",
        answer: "COBALT-2846-WILLOW",
        question: "What is the vault phrase for the east storage hall? Reply with the phrase only.",
    },
];

/// Filler sentence counts of the two calibration probes.
pub const PROBE_SENTENCES: [usize; 2] = [64, 320];

/// The user message for a needle document of `sentences` filler sentences.
/// With no needle the same frame is used to measure tokens per sentence.
pub fn needle_prompt(needle: &Needle, sentences: usize, planted: bool) -> String {
    let half = sentences / 2;
    let mut document = filler(needle.seed, half);
    document.push_str("\n\n");
    if planted {
        document.push_str(needle.fact);
        document.push_str("\n\n");
    }
    let rest: Vec<String> =
        (half..sentences).map(|index| filler_sentence(needle.seed, index as u64)).collect();
    document.push_str(&rest.join(" "));
    format!(
        "Read the document below. Exactly one sentence in it states a code.\n\n<document>\n{document}\n</document>\n\n{}",
        needle.question
    )
}

/// One JSON-mode schema and five source texts to fill it from.
#[derive(Clone, Copy, Debug)]
pub struct JsonSchemaCase {
    pub name: &'static str,
    /// The JSON Schema, sent as `response_format.json_schema.schema`.
    pub schema: &'static str,
    /// The keys, named in the prompt as well.
    pub keys: &'static str,
    pub sources: [&'static str; 5],
}

pub const JSON_CASES: [JsonSchemaCase; 10] = [
    JsonSchemaCase {
        name: "person",
        schema: r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0,"maximum":150},"role":{"type":"string"}},"required":["name","age","role"],"additionalProperties":false}"#,
        keys: "name (string), age (integer), role (string)",
        sources: [
            "Ada Park is 34 years old and works as a ship pilot.",
            "Our new hire, Ben Ortiz, turned 27 last month; he joins as a data analyst.",
            "Contact: Chloe Nguyen, age 45, head baker.",
            "Dmitri Volkov (61) is the night watchman.",
            "The speaker is Esi Mensah, 38, a civil engineer.",
        ],
    },
    JsonSchemaCase {
        name: "order",
        schema: r#"{"type":"object","properties":{"order_id":{"type":"string"},"quantity":{"type":"integer","minimum":1},"unit_price":{"type":"number"},"express":{"type":"boolean"}},"required":["order_id","quantity","unit_price","express"],"additionalProperties":false}"#,
        keys: "order_id (string), quantity (integer), unit_price (number), express (boolean)",
        sources: [
            "Order B-2210: 4 units at 12.50 each, standard shipping.",
            "Please rush order C-0093, 1 unit priced 899.00, express delivery.",
            "Order D-7781 holds 12 items at 3.25 per item; ship it the normal way.",
            "Express order E-1402: quantity 2, unit price 45.99.",
            "Order F-5550 is 7 units at 0.80 each with no express option.",
        ],
    },
    JsonSchemaCase {
        name: "weather",
        schema: r#"{"type":"object","properties":{"city":{"type":"string"},"temperature_c":{"type":"number"},"condition":{"type":"string","enum":["sunny","cloudy","rain","snow"]}},"required":["city","temperature_c","condition"],"additionalProperties":false}"#,
        keys: "city (string), temperature_c (number), condition (one of sunny, cloudy, rain, snow)",
        sources: [
            "Oslo reports minus 4 degrees Celsius with steady snow.",
            "It is 31 C and clear and sunny in Seville today.",
            "Light rain in Dublin, 11 degrees Celsius.",
            "Overcast and cloudy over Hamburg at 8.5 C.",
            "Lima: 19 degrees Celsius, sunny skies.",
        ],
    },
    JsonSchemaCase {
        name: "book",
        schema: r#"{"type":"object","properties":{"title":{"type":"string"},"author":{"type":"string"},"year":{"type":"integer"},"genres":{"type":"array","items":{"type":"string"}}},"required":["title","author","year","genres"],"additionalProperties":false}"#,
        keys: "title (string), author (string), year (integer), genres (array of strings)",
        sources: [
            "The Salt Road, a historical adventure novel by Mira Castell, came out in 2019.",
            "Published in 2004, Quiet Engines by Tomas Reyes is science fiction and mystery.",
            "Paper Gardens (Lena Holt, 2012) is a memoir.",
            "Arun Patel wrote the fantasy and romance novel Ember Tide in 2021.",
            "Cold Harbor, a 1998 thriller by Grace Lindqvist.",
        ],
    },
    JsonSchemaCase {
        name: "ticket",
        schema: r#"{"type":"object","properties":{"ticket_id":{"type":"string"},"priority":{"type":"string","enum":["low","medium","high"]},"component":{"type":"string"},"reproducible":{"type":"boolean"}},"required":["ticket_id","priority","component","reproducible"],"additionalProperties":false}"#,
        keys: "ticket_id (string), priority (one of low, medium, high), component (string), reproducible (boolean)",
        sources: [
            "Ticket T-301: the login page crashes every time; high priority.",
            "T-302 is a low priority typo in the settings screen that we can always reproduce.",
            "Medium priority ticket T-303: the export job hangs now and then, and we could not reproduce it.",
            "T-304, high priority, payments service drops requests at random and we cannot reproduce it.",
            "Ticket T-305 for the search component, low priority, reproducible on every run.",
        ],
    },
    JsonSchemaCase {
        name: "flight",
        schema: r#"{"type":"object","properties":{"flight_number":{"type":"string"},"origin":{"type":"string"},"destination":{"type":"string"},"departure_hour":{"type":"integer","minimum":0,"maximum":23}},"required":["flight_number","origin","destination","departure_hour"],"additionalProperties":false}"#,
        keys: "flight_number (string), origin (string), destination (string), departure_hour (integer 0 to 23)",
        sources: [
            "Flight VA210 leaves Lisbon for Madrid at 7 in the morning.",
            "VA388 departs Toronto at 14:00, bound for Chicago.",
            "The 22:00 departure from Tokyo to Seoul is flight VA517.",
            "VA049: Cairo to Athens, wheels up at 6.",
            "Flight VA733 from Denver to Austin departs at hour 19.",
        ],
    },
    JsonSchemaCase {
        name: "recipe",
        schema: r#"{"type":"object","properties":{"dish":{"type":"string"},"servings":{"type":"integer","minimum":1},"vegetarian":{"type":"boolean"},"ingredients":{"type":"array","items":{"type":"string"}}},"required":["dish","servings","vegetarian","ingredients"],"additionalProperties":false}"#,
        keys: "dish (string), servings (integer), vegetarian (boolean), ingredients (array of strings)",
        sources: [
            "Tomato soup for 4: tomatoes, onion, garlic and basil. No meat.",
            "Chicken curry serves 6 and needs chicken, rice, curry paste and coconut milk.",
            "A vegetarian lentil stew for 2 with lentils, carrots and cumin.",
            "Beef tacos for 3: beef, tortillas and salsa.",
            "Pancakes, 8 servings, vegetarian: flour, eggs, milk, sugar.",
        ],
    },
    JsonSchemaCase {
        name: "invoice",
        schema: r#"{"type":"object","properties":{"invoice_id":{"type":"string"},"customer":{"type":"string"},"amount_cents":{"type":"integer","minimum":0},"paid":{"type":"boolean"}},"required":["invoice_id","customer","amount_cents","paid"],"additionalProperties":false}"#,
        keys: "invoice_id (string), customer (string), amount_cents (integer), paid (boolean)",
        sources: [
            "Invoice INV-88 to Northwind Ltd for 1250 cents is still open.",
            "Harbor Foods paid invoice INV-89 in full: 49999 cents.",
            "INV-90, customer Blue Mesa Co, amount 300 cents, unpaid.",
            "Paid: invoice INV-91 from Atlas Tools, 7700 cents.",
            "Invoice INV-92 for Quill and Ink, 15 cents, not paid yet.",
        ],
    },
    JsonSchemaCase {
        name: "sensor",
        schema: r#"{"type":"object","properties":{"sensor_id":{"type":"string"},"reading":{"type":"number"},"unit":{"type":"string","enum":["celsius","percent","pascal"]},"ok":{"type":"boolean"}},"required":["sensor_id","reading","unit","ok"],"additionalProperties":false}"#,
        keys: "sensor_id (string), reading (number), unit (one of celsius, percent, pascal), ok (boolean)",
        sources: [
            "Sensor S-11 reads 21.5 degrees Celsius; status normal.",
            "Humidity sensor S-12 shows 87 percent and raised a fault.",
            "S-13 pressure: 101325 pascal, healthy.",
            "Sensor S-14 at minus 3.2 celsius, fault flag set.",
            "S-15 humidity 40 percent, all good.",
        ],
    },
    JsonSchemaCase {
        name: "meeting",
        schema: r#"{"type":"object","properties":{"title":{"type":"string"},"day":{"type":"string","enum":["monday","tuesday","wednesday","thursday","friday"]},"duration_minutes":{"type":"integer","minimum":1},"attendees":{"type":"array","items":{"type":"string"}}},"required":["title","day","duration_minutes","attendees"],"additionalProperties":false}"#,
        keys: "title (string), day (one of monday to friday), duration_minutes (integer), attendees (array of strings)",
        sources: [
            "Budget review on Monday, 60 minutes, with Ana and Raj.",
            "A 30 minute design sync on Wednesday for Kim, Leo and Sara.",
            "Friday retro, 45 minutes; attendees Omar and Lin.",
            "Hiring panel Tuesday for 90 minutes with Eve.",
            "Thursday roadmap planning: 120 minutes, Max, Ida, Tom and Uma.",
        ],
    },
];

/// The source index whose prompt names no key: only `response_format`
/// carries the schema, so a server that ignores it cannot pass these items
/// on prompt following alone.
pub const SCHEMA_ONLY_SOURCE: usize = 4;

/// The user message for JSON item `source` (an index) of `case`.
pub fn json_prompt(case: &JsonSchemaCase, source: usize) -> String {
    let text = case.sources[source];
    if source == SCHEMA_ONLY_SOURCE {
        return format!("Extract the facts from the text below as one JSON object. Reply with the JSON object only.\n\nText: {text}");
    }
    format!(
        "Extract the facts from the text below as one JSON object with exactly these keys: {}. Reply with the JSON object only.\n\nText: {text}",
        case.keys
    )
}

/// One function tool offered on every tool request.
#[derive(Clone, Copy, Debug)]
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    /// The JSON Schema of the arguments.
    pub parameters: &'static str,
}

pub const TOOLS: [Tool; 5] = [
    Tool {
        name: "get_weather",
        description: "Get the current weather for a city.",
        parameters: r#"{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city","unit"],"additionalProperties":false}"#,
    },
    Tool {
        name: "convert_currency",
        description: "Convert an amount of money from one currency to another.",
        parameters: r#"{"type":"object","properties":{"amount":{"type":"number"},"from":{"type":"string","description":"three-letter currency code"},"to":{"type":"string","description":"three-letter currency code"}},"required":["amount","from","to"],"additionalProperties":false}"#,
    },
    Tool {
        name: "set_timer",
        description: "Start a countdown timer.",
        parameters: r#"{"type":"object","properties":{"minutes":{"type":"integer"},"label":{"type":"string"}},"required":["minutes","label"],"additionalProperties":false}"#,
    },
    Tool {
        name: "search_files",
        description: "Search the project files for a text pattern.",
        parameters: r#"{"type":"object","properties":{"pattern":{"type":"string"},"max_results":{"type":"integer"}},"required":["pattern","max_results"],"additionalProperties":false}"#,
    },
    Tool {
        name: "create_event",
        description: "Create a calendar event.",
        parameters: r#"{"type":"object","properties":{"title":{"type":"string"},"day":{"type":"string","enum":["monday","tuesday","wednesday","thursday","friday"]},"hour":{"type":"integer","minimum":0,"maximum":23}},"required":["title","day","hour"],"additionalProperties":false}"#,
    },
];

/// One tool prompt and the one call it must produce.
#[derive(Clone, Copy, Debug)]
pub struct ToolCase {
    pub id: &'static str,
    pub prompt: &'static str,
    pub name: &'static str,
    /// The exact arguments as one JSON object.
    pub arguments: &'static str,
}

pub const TOOL_CASES: [ToolCase; 20] = [
    ToolCase { id: "weather-lisbon", prompt: "What is the weather in Lisbon right now? Use the unit celsius.", name: "get_weather", arguments: r#"{"city":"Lisbon","unit":"celsius"}"# },
    ToolCase { id: "weather-boston", prompt: "Check the current weather in Boston in fahrenheit.", name: "get_weather", arguments: r#"{"city":"Boston","unit":"fahrenheit"}"# },
    ToolCase { id: "weather-nairobi", prompt: "I am flying to Nairobi. Look up the weather there, in celsius.", name: "get_weather", arguments: r#"{"city":"Nairobi","unit":"celsius"}"# },
    ToolCase { id: "weather-zurich", prompt: "Get the weather for the city \"Zürich\" with unit \"fahrenheit\".", name: "get_weather", arguments: r#"{"city":"Zürich","unit":"fahrenheit"}"# },
    ToolCase { id: "fx-usd-eur", prompt: "Convert 250 USD to EUR.", name: "convert_currency", arguments: r#"{"amount":250,"from":"USD","to":"EUR"}"# },
    ToolCase { id: "fx-gbp-jpy", prompt: "How much is 19.99 GBP in JPY? Use the converter.", name: "convert_currency", arguments: r#"{"amount":19.99,"from":"GBP","to":"JPY"}"# },
    ToolCase { id: "fx-chf-usd", prompt: "Convert an amount of 1000 from CHF to USD.", name: "convert_currency", arguments: r#"{"amount":1000,"from":"CHF","to":"USD"}"# },
    ToolCase { id: "fx-eur-sek", prompt: "Please convert 75.5 EUR into SEK.", name: "convert_currency", arguments: r#"{"amount":75.5,"from":"EUR","to":"SEK"}"# },
    ToolCase { id: "timer-tea", prompt: "Set a timer for 4 minutes with the label \"tea\".", name: "set_timer", arguments: r#"{"minutes":4,"label":"tea"}"# },
    ToolCase { id: "timer-pasta", prompt: "Start a 12 minute timer labeled \"pasta\".", name: "set_timer", arguments: r#"{"minutes":12,"label":"pasta"}"# },
    ToolCase { id: "timer-standup", prompt: "Timer please: 15 minutes, label \"standup\".", name: "set_timer", arguments: r#"{"minutes":15,"label":"standup"}"# },
    ToolCase { id: "timer-laundry", prompt: "Set a timer of 45 minutes and call it \"laundry\".", name: "set_timer", arguments: r#"{"minutes":45,"label":"laundry"}"# },
    ToolCase { id: "search-todo", prompt: "Search the project files for the pattern \"TODO\" and return at most 20 results.", name: "search_files", arguments: r#"{"pattern":"TODO","max_results":20}"# },
    ToolCase { id: "search-unwrap", prompt: "Find up to 5 places in the files that match \"unwrap()\".", name: "search_files", arguments: r#"{"pattern":"unwrap()","max_results":5}"# },
    ToolCase { id: "search-fn-main", prompt: "Search the files for \"fn main\", max 3 results.", name: "search_files", arguments: r#"{"pattern":"fn main","max_results":3}"# },
    ToolCase { id: "search-license", prompt: "Look for the text \"SPDX-License-Identifier\" in the project files; I need 50 results at most.", name: "search_files", arguments: r#"{"pattern":"SPDX-License-Identifier","max_results":50}"# },
    ToolCase { id: "event-review", prompt: "Create an event titled \"Design review\" on tuesday at hour 10.", name: "create_event", arguments: r#"{"title":"Design review","day":"tuesday","hour":10}"# },
    ToolCase { id: "event-lunch", prompt: "Put \"Team lunch\" on my calendar for friday at 12.", name: "create_event", arguments: r#"{"title":"Team lunch","day":"friday","hour":12}"# },
    ToolCase { id: "event-oneonone", prompt: "Schedule an event called \"1:1 with Sam\" on monday at hour 16.", name: "create_event", arguments: r#"{"title":"1:1 with Sam","day":"monday","hour":16}"# },
    ToolCase { id: "event-deploy", prompt: "Add a calendar event \"Release deploy\" on thursday, hour 9.", name: "create_event", arguments: r#"{"title":"Release deploy","day":"thursday","hour":9}"# },
];

/// System message for tool requests.
pub const TOOL_SYSTEM: &str =
    "You can call tools. When a tool can answer the request, call exactly one tool with arguments taken from the request.";

/// Cache prompts: each is a fresh filler document, then one fixed question.
/// The cache runs send `"cache": true`, which keeps a prompt of any length.
pub const CACHE_PROMPTS: usize = 20;
pub const CACHE_SENTENCES: usize = 96;
pub fn cache_prompt(index: usize) -> String {
    format!(
        "Read the notes below.\n\n<notes>\n{}\n</notes>\n\nIn two sentences, say which places the notes mention most often.",
        filler(4_000 + index as u64, CACHE_SENTENCES)
    )
}

/// Concurrency prompts: answers long enough that eight copies decode
/// together.
pub const CONCURRENCY_PROMPTS: [(&str, &str); 3] = [
    ("hash-map", "Explain in about 120 words how a hash map handles collisions."),
    ("lighthouse", "Write a short story of about 120 words about a lighthouse keeper who finds a letter."),
    ("doc-tips", "List ten practical tips for writing clear technical documentation, one per line."),
];
pub const CONCURRENT_COPIES: usize = 8;
