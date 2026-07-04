// src/utils/guidance_grammar.rs
//! Clean-sheet grammar generation for llguidance
//! Handles constraints, tools, and reasoning in a simple, idiomatic way

use llguidance::api::TopLevelGrammar;
use once_cell::sync::Lazy;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use crate::server::parser::ToolConfig;
use crate::server::ChatCompletionRequest;
use crate::tools::Tool;
use crate::utils::chat_template::ChatTemplate;
use crate::utils::guidance::GuidanceTokens;
use tokenizers::Tokenizer;

const GRAMMAR_CACHE_MAX_ENTRIES: usize = 128;

static GRAMMAR_CACHE: Lazy<Mutex<GrammarCache>> = Lazy::new(|| Mutex::new(GrammarCache::default()));

#[derive(Default)]
struct GrammarCache {
    entries: HashMap<String, TopLevelGrammar>,
    order: VecDeque<String>,
}

impl GrammarCache {
    fn get(&self, key: &str) -> Option<TopLevelGrammar> {
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: String, grammar: TopLevelGrammar) {
        if self.entries.contains_key(&key) {
            self.entries.insert(key, grammar);
            return;
        }

        self.entries.insert(key.clone(), grammar);
        self.order.push_back(key);

        while self.entries.len() > GRAMMAR_CACHE_MAX_ENTRIES {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
    }
}

fn grammar_cache_get(key: &str) -> Option<TopLevelGrammar> {
    GRAMMAR_CACHE.lock().ok().and_then(|cache| cache.get(key))
}

fn grammar_cache_insert(key: String, grammar: TopLevelGrammar) {
    if let Ok(mut cache) = GRAMMAR_CACHE.lock() {
        cache.insert(key, grammar);
    }
}

// COMMON TRAITS

/// Common trait for grammar builders in llguidance integration
///
/// Each grammar type must implement `build_lark()` to generate its Lark representation.
/// Default implementations are provided for composition methods; override when needed.
trait GrammarBuilder: Clone + std::fmt::Debug + Sized {
    /// Build the Lark grammar string - must be implemented by each grammar type
    fn build_lark(&mut self) -> String;

    /// Compose two grammars with alternation (OR) - defaults to cloning 'other'
    /// Override when specific alternation logic is needed
    fn compose_alternate(&mut self, other: &mut Self) -> Self {
        other.clone()
    }

    /// Convert to TopLevelGrammar - defaults to parsing build_lark() output
    fn format(&mut self) -> TopLevelGrammar {
        TopLevelGrammar::from_lark_ascii(&self.build_lark())
    }
}

/// Result type for grammar-related operations
pub type GrammarResult<T> = Result<T, GrammarError>;

/// Error type for grammar-related operations
#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("Invalid grammar: {0}")]
    InvalidGrammar(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

/// Extension trait for TopLevelGrammar with built-in sanitization
/// This ensures all grammar construction paths sanitize inputs consistently
pub trait TopLevelGrammarExt: Sized {
    /// Create TopLevelGrammar from regex with ASCII sanitization
    fn from_regex_ascii(regex: &str) -> Self;

    /// Create TopLevelGrammar from Lark string with ASCII sanitization
    fn from_lark_ascii(lark: &str) -> Self;

    /// Create TopLevelGrammar from JSON schema with ASCII sanitization
    fn from_json_schema_ascii(schema: serde_json::Value) -> Result<Self, anyhow::Error>;
}

impl TopLevelGrammarExt for TopLevelGrammar {
    fn from_regex_ascii(regex: &str) -> Self {
        let sanitized = sanitize_ascii_only(regex);
        Self::from_regex(&sanitized)
    }

    fn from_lark_ascii(lark: &str) -> Self {
        let sanitized = sanitize_ascii_only(lark);
        Self::from_lark(sanitized)
    }

    fn from_json_schema_ascii(schema: serde_json::Value) -> Result<Self, anyhow::Error> {
        let schema_str = serde_json::to_string(&schema)?;
        let sanitized = sanitize_ascii_only(&schema_str);
        let val = serde_json::from_str(&sanitized)?;
        Ok(Self::from_json_schema(val))
    }
}

// UTILITY FUNCTIONS

/// Sanitize schema for llguidance - resolves $ref references and strips metadata
/// This function extracts definitions from $defs, resolves all $ref references,
/// and removes the $defs section entirely since llguidance doesn't support $ref.
fn sanitize_schema_for_llguidance_recursive(schema: &Value) -> Value {
    // Extract definitions and resolve references
    let (schema_without_defs, defs) = extract_defs(schema);

    // Resolve all $ref references in the schema
    let resolved_schema = resolve_schema_refs(&schema_without_defs, &defs);

    // Now sanitize the resolved schema (strip metadata, keep validation keywords)
    sanitize_sanitized_schema_recursive(&resolved_schema)
}

/// Sanitize a schema that has already had $refs resolved
/// This strips metadata fields like description, default, title while keeping validation keywords
fn sanitize_sanitized_schema_recursive(schema: &Value) -> Value {
    // JSON Schema validation keywords that should be KEPT
    // Based on llguidance parser/src/json/schema.rs IMPLEMENTED and META_AND_ANNOTATIONS
    const VALIDATION_KEYWORDS: &[&str] = &[
        // Core
        "anyOf",
        "oneOf",
        "allOf",
        "$ref",
        "const",
        "enum",
        "type",
        // Array
        "items",
        "additionalItems",
        "prefixItems",
        "minItems",
        "maxItems",
        // Object
        "properties",
        "additionalProperties",
        "patternProperties",
        "required",
        "minProperties",
        "maxProperties",
        // String
        "minLength",
        "maxLength",
        "pattern",
        "format",
        // Number
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        // Schema definitions (for $ref resolution)
        "$defs",
        "definitions",
        "$anchor",
    ];

    match schema {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, value) in map {
                if key == "properties" {
                    // Preserve property names (field names) - they are NOT validation keywords
                    // but we still need to process the schema values inside properties
                    if let Value::Object(props) = value {
                        let mut new_props = serde_json::Map::new();
                        for (prop_name, prop_schema) in props {
                            new_props.insert(
                                prop_name.clone(),
                                sanitize_sanitized_schema_recursive(prop_schema),
                            );
                        }
                        out.insert(key.clone(), Value::Object(new_props));
                    } else {
                        out.insert(key.clone(), sanitize_sanitized_schema_recursive(value));
                    }
                } else if VALIDATION_KEYWORDS.contains(&key.as_str()) {
                    // Keep validation keywords, strip metadata/annotation fields
                    out.insert(key.clone(), sanitize_sanitized_schema_recursive(value));
                }
                // Skip all other fields (metadata, annotations, etc.)
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(sanitize_sanitized_schema_recursive)
                .collect(),
        ),
        _ => schema.clone(),
    }
}

pub fn sanitize_schema_for_llguidance(schema: &Value) -> Value {
    sanitize_schema_for_llguidance_recursive(schema)
}

pub fn sanitize_ascii_only(s: &str) -> String {
    let mut result = String::new();
    for ch in s.chars() {
        if ch.is_ascii() {
            result.push(ch);
        }
    }
    result
}

/// Resolve $ref references by inlining the definitions from $defs
/// This is required because llguidance's JSON schema parser doesn't support $ref
fn resolve_schema_refs(schema: &Value, defs: &HashMap<String, Value>) -> Value {
    fn resolve_recursive(schema: &Value, defs: &HashMap<String, Value>) -> Value {
        match schema {
            Value::Object(map) => {
                // Check if this object is a simple $ref (single key "$ref")
                if map.len() == 1 {
                    if let Some(ref_value) = map.get("$ref") {
                        if let Value::String(ref_path) = ref_value {
                            // Handle both $defs/TypeName and #/$defs/TypeName formats
                            let def_name = ref_path
                                .strip_prefix("#/$defs/")
                                .or_else(|| ref_path.strip_prefix("$defs/"))
                                .or_else(|| ref_path.strip_prefix("#/definitions/"))
                                .or_else(|| ref_path.strip_prefix("definitions/"));

                            if let Some(name) = def_name {
                                if let Some(def) = defs.get(name) {
                                    // Found a matching definition - resolve it recursively
                                    // and return the resolved definition directly
                                    return resolve_recursive(def, defs);
                                }
                            }
                        }
                    }
                }

                // Not a simple $ref object, process all keys normally
                let mut out = serde_json::Map::new();
                for (key, value) in map {
                    if key == "$defs" || key == "definitions" {
                        // Skip definitions - they're already inlined
                        continue;
                    } else {
                        // Recursively process nested values
                        out.insert(key.clone(), resolve_recursive(value, defs));
                    }
                }
                Value::Object(out)
            }
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .map(|item| resolve_recursive(item, defs))
                    .collect(),
            ),
            _ => schema.clone(),
        }
    }

    resolve_recursive(schema, defs)
}

/// Extract $defs from schema and return (schema_without_defs, defs_map)
fn extract_defs(schema: &Value) -> (Value, HashMap<String, Value>) {
    match schema {
        Value::Object(map) => {
            let mut defs = HashMap::new();
            let mut out = serde_json::Map::new();

            for (key, value) in map {
                if key == "$defs" || key == "definitions" {
                    if let Value::Object(def_map) = value {
                        for (def_name, def_value) in def_map {
                            defs.insert(def_name.clone(), def_value.clone());
                        }
                    }
                } else {
                    out.insert(key.clone(), value.clone());
                }
            }

            (Value::Object(out), defs)
        }
        _ => (schema.clone(), HashMap::new()),
    }
}

/// Lark literal quoting - wraps string in quotes and escapes special characters
pub fn lark_quote(value: &str) -> String {
    let ascii_only = sanitize_ascii_only(value);
    serde_json::to_string(&ascii_only).unwrap_or_else(|_| "\"\"".to_string())
}

// STRUCTURED CONSTRAINTS

#[derive(Clone, Debug)]
pub enum StructuredConstraint {
    Choice(Vec<String>),
    Regex(String),
    Json(Value),
    Lark(String),
    StructuralTag(StructuralTagConfig),
}

#[derive(Clone, Debug)]
pub struct StructuralTagConfig {
    pub start_tag: String,
    pub end_tag: String,
    pub schema: Value,
}

impl StructuredConstraint {
    pub fn build_lark(&mut self) -> String {
        match self {
            StructuredConstraint::Choice(choices) => {
                let mut parts = Vec::with_capacity(choices.len());
                for choice in choices {
                    if !choice.is_empty() {
                        parts.push(lark_quote(choice));
                    }
                }
                format!("start: {}\n", parts.join(" | "))
            }
            StructuredConstraint::Regex(pattern) => format!(
                r#"start: text
text: /{}/"#,
                pattern
            ),
            StructuredConstraint::Json(schema) => {
                let sanitized = sanitize_schema_for_llguidance(schema);
                let schema_str = serde_json::to_string(&sanitized).unwrap_or_default();
                format!(
                    r#"start: text
text: %json {}"#,
                    schema_str
                )
            }
            StructuredConstraint::Lark(grammar) => grammar.clone(),
            StructuredConstraint::StructuralTag(config) => {
                let start_tag = lark_quote(&config.start_tag);
                let end_tag = lark_quote(&config.end_tag);
                format!(
                    r#"start: text
text: {} content {}
content: /[\x20-\x7E\x0A\x0D]+?/"#,
                    start_tag, end_tag
                )
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructuredOutputsGrammar {
    pub constraint: StructuredConstraint,
}

impl Default for StructuredOutputsGrammar {
    fn default() -> Self {
        Self {
            constraint: StructuredConstraint::Lark(String::new()),
        }
    }
}

impl StructuredOutputsGrammar {
    pub fn new(constraint: StructuredConstraint) -> Self {
        Self { constraint }
    }
}

impl GrammarBuilder for StructuredOutputsGrammar {
    fn build_lark(&mut self) -> String {
        self.constraint.build_lark()
    }
    fn compose_alternate(&mut self, other: &mut Self) -> Self {
        // Extract the constraint Lark strings
        let this_lark = self.build_lark();
        let other_lark = other.build_lark();

        // Extract start RHS from both (the part after "start: ")
        let this_start = this_lark
            .lines()
            .next()
            .and_then(|l| l.strip_prefix("start: "))
            .unwrap_or("text");
        let other_start = other_lark
            .lines()
            .next()
            .and_then(|l| l.strip_prefix("start: "))
            .unwrap_or("text");

        // Combine the start alternatives with repetition to allow text followed by tool_call
        let combined_start = format!("( {} | {} )+", this_start, other_start);

        // Extract non-start rules from both grammars, deduplicate and filter empty lines
        let this_rules: Vec<String> = this_lark
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty() && l.contains(':') && !l.trim().starts_with("start:"))
            .map(|s| s.trim().to_string())
            .collect();

        let other_rules: Vec<String> = other_lark
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty() && l.contains(':') && !l.trim().starts_with("start:"))
            .map(|s| s.trim().to_string())
            .collect();

        // Combine all rules and deduplicate
        let all_rules: Vec<String> = [this_rules, other_rules].concat();
        let mut seen = std::collections::HashSet::new();
        let unique_rules: Vec<String> = all_rules
            .into_iter()
            .filter(|l| {
                if seen.contains(l) {
                    false
                } else {
                    seen.insert(l.clone());
                    true
                }
            })
            .collect();

        let combined_rules = unique_rules.join("\n");

        Self {
            constraint: StructuredConstraint::Lark(format!(
                "start: {}\n{}",
                combined_start, combined_rules
            )),
        }
    }

    fn format(&mut self) -> TopLevelGrammar {
        TopLevelGrammar::from_lark_ascii(&self.build_lark())
    }
}

// TOOL CALL GRAMMAR

#[derive(Clone, Debug)]
pub enum ToolFormat {
    QwenCoder,
    MiniMax,
    Glm47Moe,
    Json,
    Generic,
}

#[derive(Clone, Debug)]
pub struct ToolCallGrammar {
    pub tools: Vec<Tool>,
    pub start_token_id: u32,
    pub end_token_id: u32,
    pub format: ToolFormat,
    marker_token_ids: HashMap<String, u32>,
    value_rules: HashMap<String, String>,
}

impl Default for ToolCallGrammar {
    fn default() -> Self {
        Self {
            tools: Vec::new(),
            start_token_id: 0,
            end_token_id: 0,
            format: ToolFormat::Json,
            marker_token_ids: HashMap::new(),
            value_rules: HashMap::new(),
        }
    }
}

impl ToolCallGrammar {
    pub fn new_generic(tools: Vec<Tool>, start_token_id: u32, end_token_id: u32) -> Self {
        Self {
            tools,
            start_token_id,
            end_token_id,
            format: ToolFormat::Generic,
            marker_token_ids: HashMap::new(),
            value_rules: HashMap::new(),
        }
    }
    pub fn new_qwen_coder(tools: Vec<Tool>, start_token_id: u32, end_token_id: u32) -> Self {
        Self {
            tools,
            start_token_id,
            end_token_id,
            format: ToolFormat::QwenCoder,
            marker_token_ids: HashMap::new(),
            value_rules: HashMap::new(),
        }
    }
    pub fn new_minimax(tools: Vec<Tool>, start_token_id: u32, end_token_id: u32) -> Self {
        Self {
            tools,
            start_token_id,
            end_token_id,
            format: ToolFormat::MiniMax,
            marker_token_ids: HashMap::new(),
            value_rules: HashMap::new(),
        }
    }
    pub fn new_glm47_moe(
        tools: Vec<Tool>,
        start_token_id: u32,
        end_token_id: u32,
        marker_token_ids: HashMap<String, u32>,
    ) -> Self {
        Self {
            tools,
            start_token_id,
            end_token_id,
            format: ToolFormat::Glm47Moe,
            marker_token_ids,
            value_rules: HashMap::new(),
        }
    }
    pub fn new_json(tools: Vec<Tool>, start_token_id: u32, end_token_id: u32) -> Self {
        Self {
            tools,
            start_token_id,
            end_token_id,
            format: ToolFormat::Json,
            marker_token_ids: HashMap::new(),
            value_rules: HashMap::new(),
        }
    }
}

impl GrammarBuilder for ToolCallGrammar {
    fn build_lark(&mut self) -> String {
        match self.format {
            ToolFormat::QwenCoder => self.build_qwen_coder_lark(),
            ToolFormat::MiniMax => self.build_minimax_lark(),
            ToolFormat::Glm47Moe => self.build_glm47_moe_lark(),
            ToolFormat::Json => self.build_json_lark(),
            ToolFormat::Generic => self.build_generic_lark(),
        }
    }
    fn compose_alternate(&mut self, _other: &mut Self) -> Self {
        self.clone()
    }
    fn format(&mut self) -> TopLevelGrammar {
        TopLevelGrammar::from_lark_ascii(&self.build_lark())
    }
}

impl ToolCallGrammar {
    pub fn build_generic_lark(&mut self) -> String {
        if self.tools.is_empty() {
            r#"start: text
 text: /(?s:.+?)/
"#
            .to_string()
        } else {
            format!(
                r#"start: tool_call
tool_call: <[{}]> text <[{}]>
text: /(?s:.+?)/
"#,
                self.start_token_id, self.end_token_id
            )
        }
    }

    fn build_json_lark(&mut self) -> String {
        let start_tag = format!("<[{}]>", self.start_token_id);
        let end_tag = format!("<[{}]>", self.end_token_id);
        let payload_schema = if self.tools.is_empty() {
            serde_json::json!({ "type": "object" })
        } else {
            let variants: Vec<Value> = self.tools.iter().map(|tool| {
                let arguments_schema = sanitize_schema_for_llguidance(&tool.function.parameters);
                serde_json::json!({
                    "type": "object",
                    "properties": { "name": { "type": "string", "enum": [tool.function.name.clone()] }, "arguments": arguments_schema },
                    "required": ["name", "arguments"], "additionalProperties": false,
                })
            }).collect();
            if variants.len() == 1 {
                variants[0].clone()
            } else {
                serde_json::json!({ "oneOf": variants })
            }
        };
        let payload_schema_str = serde_json::to_string(&payload_schema).unwrap_or_default();
        format!(
            r#"start: tool_call
 tool_call: {} tool_content {}
 tool_content: %json {}"#,
            start_tag, end_tag, payload_schema_str
        )
    }

    fn build_value_rules(&self) -> Vec<String> {
        let mut rules: Vec<String> = Vec::new();
        let mut sorted_rules: Vec<_> = self.value_rules.iter().collect();
        sorted_rules.sort_by(|a, b| a.0.cmp(b.0));

        for (rule_name, pattern) in sorted_rules {
            // Check if pattern already contains the LHS (multi-line pattern)
            // If pattern starts with "rule_name: ", don't add another "rule_name: "
            let output = if pattern.starts_with(&format!("{}: ", rule_name)) {
                pattern.clone()
            } else {
                format!("{}: {}", rule_name, pattern)
            };
            rules.push(output);
        }
        rules
    }

    fn build_qwen_coder_lark(&mut self) -> String {
        let mut rules: Vec<String> = Vec::new();
        let envelope_start_tag = format!("<[{}]>", self.start_token_id);
        let envelope_end_tag = format!("<[{}]>", self.end_token_id);
        let tool_rule_names: Vec<String> = (0..self.tools.len())
            .map(|i| format!("tool_{}", i))
            .collect();
        rules.push("start: tool_call".to_string());
        rules.push(format!(
            r#"tool_call: {} tool_content {} "#,
            envelope_start_tag, envelope_end_tag
        ));
        let tools = self.tools.clone();
        for (tool_idx, tool) in tools.iter().enumerate() {
            let tool_name_ascii: String = tool
                .function
                .name
                .chars()
                .filter(|c| c.is_ascii())
                .collect();
            let func_end = lark_quote("</function>\n");
            if let Some(props) = tool
                .function
                .parameters
                .get("properties")
                .and_then(|p| p.as_object())
            {
                let mut param_rules_vec: Vec<String> = Vec::new();
                for (param_idx, (param_name, schema)) in props.iter().enumerate() {
                    let param_name_ascii: String =
                        param_name.chars().filter(|c| c.is_ascii()).collect();
                    let param_tag = lark_quote(&format!("\n<parameter={}>\n", param_name_ascii));
                    let param_end = lark_quote("\n</parameter>\n");
                    let param_rule = format!("param_{}_{}", tool_idx, param_idx);
                    let param_type = schema
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("string")
                        .to_string();
                    let value_rule =
                        self.get_value_rule_name(tool_idx, param_idx, &param_type, schema);
                    if param_type == "string" {
                        rules.push(format!(r#"{}: {} {} "#, param_rule, param_tag, value_rule));
                    } else {
                        rules.push(format!(
                            r#"{}: {} {} {} "#,
                            param_rule, param_tag, value_rule, param_end
                        ));
                    }
                    let required_params: Vec<String> = tool
                        .function
                        .parameters
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();
                    if required_params.contains(param_name) {
                        param_rules_vec.push(format!(" {}", param_rule));
                    } else {
                        param_rules_vec.push(format!("({})?", param_rule));
                    }
                }
                let params_expr = param_rules_vec.join(" ");
                if param_rules_vec.len() > 0 {
                    let func_start = lark_quote(&format!("\n<function={}>", tool_name_ascii));
                    rules.push(format!(
                        r#"tool_{}: {}{} {}"#,
                        tool_idx, func_start, params_expr, func_end
                    ));
                } else {
                    let func_start = lark_quote(&format!("\n<function={}>\n", tool_name_ascii));
                    rules.push(format!(
                        r#"tool_{}: {}{} {}"#,
                        tool_idx, func_start, params_expr, func_end
                    ));
                }
            } else {
                let func_start = lark_quote(&format!("\n<function={}>\n", tool_name_ascii));
                rules.push(format!(r#"tool_{}: {} {}"#, tool_idx, func_start, func_end));
            }
        }
        let tool_variants = tool_rule_names.join(" | ");
        rules.push(format!("tool_content: {}", tool_variants));
        let value_rules = self.build_value_rules();
        rules.extend(value_rules);
        let lark = rules.join("\n") + "\n";
        lark
    }

    fn build_minimax_lark(&mut self) -> String {
        let mut rules: Vec<String> = Vec::new();
        let envelope_start_tag = format!("<[{}]>", self.start_token_id);
        let envelope_end_tag = format!("<[{}]>", self.end_token_id);
        let tool_rule_names: Vec<String> = (0..self.tools.len())
            .map(|i| format!("tool_{}", i))
            .collect();
        rules.push("start: tool_call".to_string());
        rules.push(format!(
            r#"tool_call: {} tool_content {} "#,
            envelope_start_tag, envelope_end_tag
        ));
        let tools = self.tools.clone();
        for (tool_idx, tool) in tools.iter().enumerate() {
            let tool_name_ascii: String = tool
                .function
                .name
                .chars()
                .filter(|c| c.is_ascii())
                .collect();
            let func_end = lark_quote("</invoke>\n");
            if let Some(props) = tool
                .function
                .parameters
                .get("properties")
                .and_then(|p| p.as_object())
            {
                let mut param_rules_vec: Vec<String> = Vec::new();
                for (param_idx, (param_name, schema)) in props.iter().enumerate() {
                    let param_name_ascii: String =
                        param_name.chars().filter(|c| c.is_ascii()).collect();
                    let param_tag = format!(r#""\n<parameter name=\"{}\">""#, param_name_ascii);
                    let param_end = lark_quote("\n</parameter>\n");
                    let param_rule = format!("param_{}_{}", tool_idx, param_idx);
                    let param_type = schema
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("string")
                        .to_string();
                    let value_rule =
                        self.get_value_rule_name(tool_idx, param_idx, &param_type, schema);
                    if param_type == "string" {
                        rules.push(format!(r#"{}: {} {} "#, param_rule, param_tag, value_rule));
                    } else {
                        rules.push(format!(
                            r#"{}: {} {} {} "#,
                            param_rule, param_tag, value_rule, param_end
                        ));
                    }
                    let required_params: Vec<String> = tool
                        .function
                        .parameters
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();
                    if required_params.contains(param_name) {
                        param_rules_vec.push(format!(" {}", param_rule));
                    } else {
                        param_rules_vec.push(format!(" ({})?", param_rule));
                    }
                }
                let params_expr = param_rules_vec.join(" ");
                if param_rules_vec.len() > 0 {
                    let func_start = format!(r#""\n<invoke name=\"{}\">""#, tool_name_ascii);
                    rules.push(format!(
                        r#"tool_{}: {}{} {}"#,
                        tool_idx, func_start, params_expr, func_end
                    ));
                } else {
                    let func_start = format!(r#""\n<invoke name=\"{}\">\n""#, tool_name_ascii);
                    rules.push(format!(
                        r#"tool_{}: {}{} {}"#,
                        tool_idx, func_start, params_expr, func_end
                    ));
                }
            } else {
                let func_start = format!(r#""\n<invoke name=\"{}\">\n""#, tool_name_ascii);
                rules.push(format!(r#"tool_{}: {} {}"#, tool_idx, func_start, func_end));
            }
        }
        let tool_variants = tool_rule_names.join(" | ");
        rules.push(format!("tool_content: {}", tool_variants));
        let value_rules = self.build_value_rules();
        rules.extend(value_rules);
        let lark = rules.join("\n") + "\n";
        lark
    }

    fn build_glm47_moe_lark(&mut self) -> String {
        let mut rules: Vec<String> = Vec::new();
        let envelope_start_tag = format!("<[{}]>", self.start_token_id);
        let envelope_end_tag = format!("<[{}]>", self.end_token_id);
        let arg_key_start = self.marker_terminal("<arg_key>");
        let arg_key_end = self.marker_terminal("</arg_key>");
        let arg_value_start = self.marker_terminal("<arg_value>");
        let arg_value_end = self.marker_terminal("</arg_value>");
        let tool_rule_names: Vec<String> = (0..self.tools.len())
            .map(|i| format!("tool_{}", i))
            .collect();
        rules.push("start: tool_call".to_string());
        rules.push(format!(
            r#"tool_call: {} tool_content {} "#,
            envelope_start_tag, envelope_end_tag
        ));
        let tools = self.tools.clone();
        for (tool_idx, tool) in tools.iter().enumerate() {
            let tool_name_ascii: String = tool
                .function
                .name
                .chars()
                .filter(|c| c.is_ascii())
                .collect();
            if let Some(props) = tool
                .function
                .parameters
                .get("properties")
                .and_then(|p| p.as_object())
            {
                let mut param_rules_vec: Vec<String> = Vec::new();
                let required_count = tool
                    .function
                    .parameters
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|arr| arr.len())
                    .unwrap_or_default()
                    .min(props.len());
                for param_idx in 0..props.len() {
                    if param_idx < required_count {
                        param_rules_vec.push(" glm_arg_pair".to_string());
                    } else {
                        param_rules_vec.push(" (glm_arg_pair)?".to_string());
                    }
                }
                rules.push(format!(
                    r#"tool_{}: {}{}"#,
                    tool_idx,
                    lark_quote(&tool_name_ascii),
                    param_rules_vec.join(" ")
                ));
            } else {
                rules.push(format!(
                    r#"tool_{}: {}"#,
                    tool_idx,
                    lark_quote(&tool_name_ascii)
                ));
            }
        }
        let tool_variants = tool_rule_names.join(" | ");
        rules.push(format!("tool_content: {}", tool_variants));
        rules.push(format!(
            r#"glm_arg_pair: {} glm_arg_key {} {} glm_arg_value? {}"#,
            arg_key_start, arg_key_end, arg_value_start, arg_value_end
        ));
        let mut key_values: Vec<String> = tools
            .iter()
            .filter_map(|tool| {
                tool.function
                    .parameters
                    .get("properties")
                    .and_then(|p| p.as_object())
            })
            .flat_map(|props| props.keys())
            .map(|key| key.chars().filter(|c| c.is_ascii()).collect::<String>())
            .filter(|key| !key.is_empty())
            .map(|key| lark_quote(&key))
            .collect();
        key_values.sort();
        key_values.dedup();
        if key_values.is_empty() {
            rules.push("glm_arg_key: /[A-Za-z0-9_.-]+/".to_string());
        } else {
            rules.push(format!("glm_arg_key: {}", key_values.join(" | ")));
        }
        if let Some(arg_value_end_id) = self.marker_token_ids.get("</arg_value>") {
            let excluded = if self.end_token_id != 0 && self.end_token_id != *arg_value_end_id {
                format!("{},{}", arg_value_end_id, self.end_token_id)
            } else {
                arg_value_end_id.to_string()
            };
            rules.push(format!("glm_arg_value: <[^{}]>+", excluded));
        } else {
            rules.push(r#"glm_arg_value[suffix="</arg_value>"]: /(?s:.+?)/"#.to_string());
        }
        rules.join("\n") + "\n"
    }

    fn marker_terminal(&self, marker: &str) -> String {
        self.marker_token_ids
            .get(marker)
            .map(|id| format!("<[{}]>", id))
            .unwrap_or_else(|| lark_quote(marker))
    }

    fn get_value_rule_name(
        &mut self,
        tool_idx: usize,
        param_idx: usize,
        param_type: &str,
        param_schema: &Value,
    ) -> String {
        let rule_name = if param_type == "string" {
            "value_string".to_string()
        } else {
            format!("value_{}_{}_{}", tool_idx, param_idx, param_type)
        };
        let pattern = if param_type == "string" {
            r#"/[\x20-\x7E\x0A\x0D]+?/"#.to_string()
        } else {
            let sanitized = sanitize_schema_for_llguidance(param_schema);
            let schema_json = serde_json::to_string(&sanitized).unwrap_or_default();
            format!("%json {}", schema_json)
        };
        let lhs = if param_type == "string" {
            match self.format {
                ToolFormat::MiniMax => {
                    format!(r#"{}[suffix="</parameter>\n"]"#, rule_name)
                }
                ToolFormat::Glm47Moe => {
                    format!(r#"{}[suffix="</arg_value>"]"#, rule_name)
                }
                _ => {
                    format!(r#"{}[suffix="\n</parameter>\n"]"#, rule_name)
                }
            }
        } else {
            rule_name.clone()
        };
        self.value_rules.insert(lhs, pattern);
        rule_name
    }
}

pub struct GrammarRequestDispatcher<'a> {
    pub request: &'a ChatCompletionRequest,
    pub guidance_tokens: &'a GuidanceTokens,
    pub tool_config: &'a crate::server::parser::ToolConfig,
    pub enable_tool_grammar: bool,
    pub parser_name: String,
    pub tokenizer: &'a Tokenizer,
    pub chat_template: Option<crate::utils::chat_template::ChatTemplate>,
    pub disable_reasoning: bool,
}

pub fn request_has_structured_constraint(request: &ChatCompletionRequest) -> bool {
    request.structured_outputs.as_ref().is_some_and(|so| {
        so.choice
            .as_ref()
            .is_some_and(|choices| !choices.is_empty())
            || so.regex.is_some()
            || so.json.is_some()
            || so.grammar.is_some()
            || so.structural_tag.is_some()
    }) || request
        .response_format
        .as_ref()
        .is_some_and(|rf| matches!(rf.format_type.as_str(), "json_schema" | "json_object"))
        || request.constraint.is_some()
}

pub fn request_has_tool_grammar(
    request: &ChatCompletionRequest,
    enable_tool_grammar: bool,
) -> bool {
    enable_tool_grammar
        && !matches!(
            request.tool_choice.as_ref(),
            Some(crate::tools::ToolChoice::Mode(
                crate::tools::ToolChoiceMode::None
            ))
        )
        && request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
}

fn request_requires_tool_call(request: &ChatCompletionRequest) -> bool {
    matches!(
        request.tool_choice.as_ref(),
        Some(crate::tools::ToolChoice::Mode(
            crate::tools::ToolChoiceMode::Required
        )) | Some(crate::tools::ToolChoice::Function { .. })
    )
}

impl<'a> GrammarRequestDispatcher<'a> {
    pub fn new(
        request: &'a ChatCompletionRequest,
        guidance_tokens: &'a GuidanceTokens,
        tool_config: &'a crate::server::parser::ToolConfig,
        enable_tool_grammar: bool,
        parser_name: String,
        tokenizer: &'a Tokenizer,
        chat_template: Option<crate::utils::chat_template::ChatTemplate>,
        disable_reasoning: bool,
    ) -> Self {
        Self {
            request,
            guidance_tokens,
            tool_config,
            enable_tool_grammar,
            parser_name,
            tokenizer,
            chat_template,
            disable_reasoning,
        }
    }

    pub fn build_grammar(self) -> Option<TopLevelGrammar> {
        let has_constraint = request_has_structured_constraint(self.request);
        let should_build_tool_grammar =
            request_has_tool_grammar(self.request, self.enable_tool_grammar);

        // Avoid request-time grammar work for ordinary tool requests unless tool
        // grammar is enabled. Explicit structured output constraints still use
        // guided decoding even when tool grammar is disabled.
        if !has_constraint && !should_build_tool_grammar {
            return None;
        }

        let cache_key = self.cache_key();
        if let Some(grammar) = grammar_cache_get(&cache_key) {
            crate::log_info!("[llg] Grammar cache hit");
            return Some(grammar);
        }

        let constraint_grammar = if has_constraint {
            self.build_constraint_grammar()
        } else {
            None
        };
        let tool_grammar = if should_build_tool_grammar {
            self.build_tool_grammar()
        } else {
            None
        };

        // vLLM/SGLang architecture: grammar constraints NEVER include reasoning.
        // Reasoning is handled at the mask level (GuidanceState defers grammar
        // masks until after the </think> token). The grammar only constrains the
        // structured output — tool call JSON, JSON schema, regex, etc.
        // Reasoning effort is used only for non-grammar reasoning control.

        // Only activate LLG when the request actually specifies something to constrain.
        if constraint_grammar.is_none() && tool_grammar.is_none() {
            return None;
        }

        let max_tokens = self.request.max_tokens.unwrap_or(0);

        let force_tool_call = request_requires_tool_call(self.request);
        let grammar = match (constraint_grammar, tool_grammar) {
            (None, Some(mut tool_grammar)) if force_tool_call => {
                StructuredOutputsGrammar::new(StructuredConstraint::Lark(tool_grammar.build_lark()))
            }
            (None, Some(tool_grammar)) => {
                let text_grammar = StructuredOutputsGrammar::new(StructuredConstraint::Lark(
                    "start: text\ntext[stop=\"\"]: /(?s:.+?)/".to_string(),
                ));
                GrammarComposer::compose_constraint_with_tools(text_grammar, Some(tool_grammar))
            }
            (constraint_grammar, tool_grammar) => {
                // Build only the structured output constraint grammar — NO reasoning wrapping.
                let constraint_grammar = constraint_grammar.unwrap_or_else(|| {
                    StructuredOutputsGrammar::new(StructuredConstraint::Lark(
                        "start: text\ntext[stop=\"\"]: /(?s:.+?)/".to_string(),
                    ))
                });

                GrammarComposer::compose_constraint_with_tools(constraint_grammar, tool_grammar)
            }
        };

        let grammar = GrammarComposer::compose_all_grammars(
            vec![grammar],
            None,
            self.guidance_tokens,
            max_tokens,
            self.chat_template,
            self.tokenizer,
        );
        grammar_cache_insert(cache_key, grammar.clone());
        Some(grammar)
    }

    fn cache_key(&self) -> String {
        let glm_marker_token_ids = if self.parser_name == "glm47_moe" {
            self.resolve_glm_marker_token_ids()
        } else {
            HashMap::new()
        };
        let key = serde_json::json!({
            "version": 2,
            "enable_tool_grammar": self.enable_tool_grammar,
            "parser_name": &self.parser_name,
            "max_tokens": self.request.max_tokens.unwrap_or(0),
            "tools": &self.request.tools,
            "tool_choice": &self.request.tool_choice,
            "structured_outputs": &self.request.structured_outputs,
            "response_format": &self.request.response_format,
            "constraint": &self.request.constraint,
            "constraint_type": &self.request.constraint_type,
            "disable_reasoning": self.disable_reasoning,
            "glm_marker_token_ids": glm_marker_token_ids,
            "guidance_tokens": {
                "bos": &self.guidance_tokens.bos_token_ids,
                "eos": &self.guidance_tokens.eos_token_ids,
                "reasoning_start": &self.guidance_tokens.reasoning_start_ids,
                "reasoning_end": &self.guidance_tokens.reasoning_end_ids,
                "tool_start": &self.guidance_tokens.tool_call_start_ids,
                "tool_end": &self.guidance_tokens.tool_call_end_ids,
                "add_bos": self.guidance_tokens.add_bos_token,
            },
            "chat_template": format!("{:?}", self.chat_template),
        });
        serde_json::to_string(&key).unwrap_or_else(|_| format!("{:?}", key))
    }

    fn token_id_for_text(tokenizer: &Tokenizer, text: &str) -> Option<u32> {
        tokenizer
            .encode(text, false)
            .ok()
            .and_then(|encoded| {
                let ids = encoded.get_ids();
                if ids.len() == 1 {
                    Some(ids[0])
                } else {
                    None
                }
            })
            .or_else(|| tokenizer.get_vocab(true).get(text).copied())
    }

    fn resolve_glm_marker_token_ids(&self) -> HashMap<String, u32> {
        ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"]
            .into_iter()
            .filter_map(|marker| {
                Self::token_id_for_text(self.tokenizer, marker).map(|id| (marker.to_string(), id))
            })
            .collect()
    }

    fn build_constraint_grammar(&self) -> Option<StructuredOutputsGrammar> {
        if let Some(ref so) = self.request.structured_outputs {
            if let Some(choice) = &so.choice {
                if !choice.is_empty() {
                    return Some(StructuredOutputsGrammar::new(StructuredConstraint::Choice(
                        choice.clone(),
                    )));
                }
            }
            if let Some(ref regex) = so.regex {
                return Some(StructuredOutputsGrammar::new(StructuredConstraint::Regex(
                    regex.clone(),
                )));
            }
            if let Some(ref json) = so.json {
                return Some(StructuredOutputsGrammar::new(StructuredConstraint::Json(
                    json.clone(),
                )));
            }
            if let Some(ref grammar) = so.grammar {
                return Some(StructuredOutputsGrammar::new(StructuredConstraint::Lark(
                    grammar.clone(),
                )));
            }
            if let Some(ref structural_tag) = so.structural_tag {
                return Some(self.build_structural_tag_grammar(structural_tag));
            }
        }
        if let Some(ref rf) = self.request.response_format {
            match rf.format_type.as_str() {
                "json_schema" => {
                    if let Some(ref schema) = rf.json_schema {
                        let schema = sanitize_schema_for_llguidance(&schema.schema);
                        return Some(StructuredOutputsGrammar::new(StructuredConstraint::Json(
                            schema,
                        )));
                    }
                }
                "json_object" => {
                    return Some(StructuredOutputsGrammar::new(StructuredConstraint::Lark(
                        "start: text\ntext: %json {\"type\":\"object\"}".to_string(),
                    )));
                }
                _ => {}
            }
        }
        if let Some(ref constraint) = self.request.constraint {
            let constraint_type = self.request.constraint_type.as_deref().unwrap_or("regex");
            match constraint_type {
                "regex" => {
                    return Some(StructuredOutputsGrammar::new(StructuredConstraint::Regex(
                        constraint.clone(),
                    )));
                }
                "lark" => {
                    return Some(StructuredOutputsGrammar::new(StructuredConstraint::Lark(
                        constraint.clone(),
                    )));
                }
                "json_schema" | "json" => {
                    if let Ok(schema) = serde_json::from_str(constraint) {
                        let schema = sanitize_schema_for_llguidance(&schema);
                        return Some(StructuredOutputsGrammar::new(StructuredConstraint::Json(
                            schema,
                        )));
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn build_structural_tag_grammar(&self, structural_tag: &Value) -> StructuredOutputsGrammar {
        let start_tag = structural_tag
            .get("start_tag")
            .or_else(|| structural_tag.get("tag"))
            .and_then(|v| v.as_str())
            .unwrap_or("<tool>")
            .to_string();
        let end_tag = structural_tag
            .get("end_tag")
            .and_then(|v| v.as_str())
            .unwrap_or("</tool>")
            .to_string();
        let schema = structural_tag
            .get("schema")
            .cloned()
            .unwrap_or(serde_json::json!({"type": "object"}));
        StructuredOutputsGrammar::new(StructuredConstraint::StructuralTag(StructuralTagConfig {
            start_tag,
            end_tag,
            schema,
        }))
    }

    fn first_tool_token_id(config_ids: &std::collections::HashSet<u32>, fallback: &[u32]) -> u32 {
        config_ids
            .iter()
            .copied()
            .min()
            .or_else(|| fallback.first().copied())
            .unwrap_or(0)
    }

    fn build_tool_grammar(&self) -> Option<ToolCallGrammar> {
        if self.request.tools.is_none() {
            return None;
        }

        let tools = self.request.tools.as_ref().unwrap().clone();
        let start_token_id = Self::first_tool_token_id(
            &self.tool_config.start_token_ids,
            &self.guidance_tokens.tool_call_start_ids,
        );
        let end_token_id = Self::first_tool_token_id(
            &self.tool_config.end_token_ids,
            &self.guidance_tokens.tool_call_end_ids,
        );

        // TODO align 1:1 with parser selection
        match self.parser_name.as_str() {
            "qwen_coder" => Some(ToolCallGrammar::new_qwen_coder(
                tools,
                start_token_id,
                end_token_id,
            )),
            "minimax_m2" => Some(ToolCallGrammar::new_minimax(
                tools,
                start_token_id,
                end_token_id,
            )),
            "glm47_moe" => Some(ToolCallGrammar::new_glm47_moe(
                tools,
                start_token_id,
                end_token_id,
                self.resolve_glm_marker_token_ids(),
            )),
            "gemma4" => Some(ToolCallGrammar::new_json(
                tools,
                start_token_id,
                end_token_id,
            )),
            "qwen" | "json" | _ => Some(ToolCallGrammar::new_json(
                tools,
                start_token_id,
                end_token_id,
            )),
        }
    }
}

// GRAMMAR COMPOSER

pub struct GrammarComposer;

impl GrammarComposer {
    pub fn compose_all_grammars(
        constraint_grammars: Vec<StructuredOutputsGrammar>,
        tool_grammar: Option<ToolCallGrammar>,
        guidance_tokens: &GuidanceTokens,
        max_tokens: usize,
        chat_template: Option<crate::utils::chat_template::ChatTemplate>,
        tokenizer: &Tokenizer,
    ) -> TopLevelGrammar {
        let merged_constraints = Self::merge_constraints(constraint_grammars);
        let composed_with_tools =
            Self::compose_constraint_with_tools(merged_constraints, tool_grammar);
        let mut grammar = Self::finalize_with_eos(composed_with_tools, guidance_tokens);

        // Derive role from chat template: MiniMax uses "ai", most others use "assistant"
        let role = chat_template
            .as_ref()
            .and_then(|t| t.get_template_string())
            .and_then(|tmpl| {
                if tmpl.contains("\"ai\"") || tmpl.contains("'ai'") {
                    Some("ai".to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "assistant".to_string());

        if guidance_tokens.add_bos_token {
            grammar = Self::prefix_with_bos(grammar, guidance_tokens, role);
        }

        // Apply thinking fallback transformation after all composition is complete
        // This transforms <[token_id]> syntax to string literals for models without reasoning tokens
        grammar = apply_thinking_fallback(grammar, guidance_tokens, chat_template, tokenizer);

        // Set max_tokens on the grammar
        grammar.max_tokens = Some(max_tokens);
        grammar
    }

    fn merge_constraints(grammars: Vec<StructuredOutputsGrammar>) -> StructuredOutputsGrammar {
        if grammars.is_empty() {
            // Default text grammar when no constraints specified
            return StructuredOutputsGrammar::new(StructuredConstraint::Lark(
                "start: text\ntext[stop=\"\"]: /(?s:.+?)/".to_string(),
            ));
        }
        if grammars.len() == 1 {
            return grammars.into_iter().next().unwrap();
        }
        // Clone grammars to avoid consuming them
        let grammars_clone = grammars.clone();
        let mut result = grammars_clone.into_iter().next().unwrap();
        let grammars_for_loop = grammars.into_iter().skip(1).collect::<Vec<_>>();
        for mut g in grammars_for_loop {
            result = result.compose_alternate(&mut g);
        }
        result
    }

    fn compose_constraint_with_tools(
        base: StructuredOutputsGrammar,
        tool: Option<ToolCallGrammar>,
    ) -> StructuredOutputsGrammar {
        match tool {
            Some(mut tool_gram) => {
                let tool_constraint = StructuredConstraint::Lark(tool_gram.build_lark());
                let mut tool_grammar = StructuredOutputsGrammar::new(tool_constraint);
                let mut base_mut = base;
                base_mut.compose_alternate(&mut tool_grammar)
            }
            None => base,
        }
    }

    fn prefix_with_bos(
        grammar: TopLevelGrammar,
        guidance_tokens: &GuidanceTokens,
        role: String,
    ) -> TopLevelGrammar {
        if guidance_tokens.bos_token_ids.is_empty() {
            return grammar;
        }

        // Check if grammar already has bos rule - avoid duplication
        let lark = get_lark_from_top_level_grammar(&grammar);
        if lark.contains("bos") {
            return grammar;
        }

        // Extract current start rule RHS
        let first_line = lark.lines().next().unwrap_or("");
        let current_start_rhs = if let Some(rhs) = first_line.strip_prefix("start: ") {
            rhs.trim()
        } else {
            "text"
        };

        // Build BOS rule(s) - support multiple BOS tokens with alternation
        let bos_rule = if guidance_tokens.bos_token_ids.len() == 1 {
            format!(
                r#"bos: <[{}]> "{}:" "\n" "#,
                guidance_tokens.bos_token_ids[0], &role
            )
        } else {
            let ids: Vec<String> = guidance_tokens
                .bos_token_ids
                .iter()
                .map(|id| format!("<[{}]>", id))
                .collect();
            format!(r#"bos: ( {} ) "{}:" "\n" "#, ids.join(" | "), &role)
        };

        // Construct new grammar with BOS prefix
        // Remove old start line, keep other rules, add new start and bos
        let remaining_rules: Vec<String> = lark
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect();

        let new_lark = format!(
            "start: bos {}\n{}\n{}",
            current_start_rhs,
            remaining_rules.join("\n"),
            bos_rule
        );

        TopLevelGrammar::from_lark_ascii(&new_lark)
    }

    fn finalize_with_eos(
        mut grammar: StructuredOutputsGrammar,
        guidance_tokens: &GuidanceTokens,
    ) -> TopLevelGrammar {
        let lark = grammar.build_lark();
        let eos_token_ids: Vec<u32> = guidance_tokens
            .eos_token_ids
            .iter()
            .copied()
            .filter(|id| {
                !guidance_tokens.tool_call_end_ids.contains(id)
                    || !lark.contains(&format!("<[{}]>", id))
            })
            .collect();

        if eos_token_ids.is_empty() {
            return grammar.format();
        }
        if lark.contains("eos") {
            return grammar.format();
        }
        let first_line = lark.lines().next().unwrap_or("");
        let current_start_rhs = if let Some(rhs) = first_line.strip_prefix("start: ") {
            rhs.trim()
        } else {
            "text"
        };
        let new_start = format!("start: {} eos", current_start_rhs);
        let eos_rule = if eos_token_ids.len() == 1 {
            format!("eos: <[{}]>", eos_token_ids[0])
        } else {
            let ids: Vec<String> = eos_token_ids
                .iter()
                .map(|id| format!("<[{}]>", id))
                .collect();
            format!("eos: ( {} )", ids.join(" | "))
        };
        let final_lark = format!(
            "{}\n{}\n{}",
            new_start,
            lark.lines().skip(1).collect::<Vec<_>>().join("\n"),
            eos_rule
        );
        TopLevelGrammar::from_lark_ascii(&final_lark)
    }
}

// HELPER FUNCTIONS

pub fn get_lark_from_top_level_grammar(grammar: &TopLevelGrammar) -> String {
    if grammar.grammars.is_empty() {
        return "No grammars".to_string();
    }
    let mut larks: Vec<String> = grammar
        .grammars
        .iter()
        .filter_map(|g| g.lark_grammar.as_ref())
        .map(|s| s.clone())
        .collect();
    for g in &grammar.grammars {
        if let Some(json_schema) = &g.json_schema {
            let schema_str = serde_json::to_string(json_schema).unwrap_or_default();
            larks.push(format!("start: text\ntext: %json {}", schema_str));
        }
    }
    if larks.is_empty() {
        format!(
            "{} grammars, none have lark_grammar",
            grammar.grammars.len()
        )
    } else {
        larks.join("\n---\n")
    }
}

/// Apply thinking fallback transformation for models without reasoning tokens in chat template
/// This version works on Lark strings for use during grammar composition
///
/// This function transforms <[token_id]> syntax to string literals like "thinking" and "</thinking>"
/// for models that were not trained on reasoning tokens and cannot properly handle the <[token_id]> syntax.
///
/// The fallback is controlled by the VLLM_RS_PROVIDE_THINKING_FALLBACK environment variable.
/// When set to true, models without explicit reasoning tokens in their chat template will have
/// their grammar transformed to use string literals instead of token IDs.
///
/// Returns Some(transformed_lark) if fallback should be applied, None otherwise
///
/// This function checks:
/// 1. If VLLM_RS_PROVIDE_THINKING_FALLBACK is set to "true" or "1"
/// 2. If the chat template contains reasoning tokens
/// 3. If the grammar contains <[token_id]> syntax that needs to be replaced
///
/// Returns None if:
/// - Environment variable is not set (fallback not enabled)
/// - Chat template already contains reasoning tokens (no need for fallback)
pub fn apply_thinking_fallback_lark(
    lark: String,
    guidance_tokens: &GuidanceTokens,
    chat_template: Option<crate::utils::chat_template::ChatTemplate>,
    tokenizer: &Tokenizer,
) -> Option<String> {
    // Check environment variable - if not set, fallback is not enabled
    let provide_thinking_fallback = std::env::var("VLLM_RS_PROVIDE_THINKING_FALLBACK")
        .ok()
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false);

    if !provide_thinking_fallback {
        return None; // Fallback not enabled via environment variable
    }

    // Get reasoning token strings from tokenizer
    match get_reasoning_token_strings(guidance_tokens, tokenizer) {
        Some((start_str, end_str)) => {
            // Check if chat template already contains reasoning tokens
            if let Some(template_str) = chat_template
                .as_ref()
                .and_then(|t| t.get_template_string().map(|s| s.to_string()))
            {
                // Normalize to ASCII-only for robust comparison
                let normalized_template: String =
                    template_str.chars().filter(|c| c.is_ascii()).collect();
                let normalized_start: String = start_str.chars().filter(|c| c.is_ascii()).collect();
                let normalized_end: String = end_str.chars().filter(|c| c.is_ascii()).collect();

                // Check if template contains reasoning tokens
                if normalized_template.contains(&normalized_start)
                    && normalized_template.contains(&normalized_end)
                {
                    crate::log_info!(
                        "[llg] Chat template contains reasoning tokens, no fallback needed"
                    );
                    return None; // Reasoning tokens found in template, no fallback needed
                }
            }

            // Apply fallback transformation
            crate::log_info!(
                "[llg] Chat template does not contain reasoning tokens, applying fallback"
            );

            let reason_start = format!("<[{}]>", guidance_tokens.reasoning_start_ids[0]);
            let reason_end = format!("<[{}]>", guidance_tokens.reasoning_end_ids[0]);

            // Transform <[token_id]> syntax to string literals in common vocabulary
            let lark = lark
                .replace(&reason_start, "\"<thinking>\"")
                .replace(&reason_end, "\"</thinking>\"");

            Some(lark)
        }
        None => {
            // No reasoning tokens found via tokenizer decode — skip fallback to avoid panic
            crate::log_info!(
                "[llg] No reasoning tokens decodable from guidance_tokens, skipping fallback"
            );
            None
        }
    }
}

/// Apply thinking fallback transformation for models without reasoning tokens in chat template
///
/// This function transforms <[token_id]> syntax to string literals like "thinking" and "</thinking>"
/// for models that were not trained on reasoning tokens and cannot properly handle the <[token_id]> syntax.
///
/// The fallback is controlled by the VLLM_RS_PROVIDE_THINKING_FALLBACK environment variable.
/// When set to true, models without explicit reasoning tokens in their chat template will have
/// their grammar transformed to use string literals instead of token IDs.
pub fn apply_thinking_fallback(
    grammar: TopLevelGrammar,
    guidance_tokens: &GuidanceTokens,
    chat_template: Option<crate::utils::chat_template::ChatTemplate>,
    tokenizer: &Tokenizer,
) -> TopLevelGrammar {
    // Extract Lark string from grammar
    let lark_str = get_lark_from_top_level_grammar(&grammar);

    // Apply the lark-based fallback transformation
    if let Some(transformed_lark) =
        apply_thinking_fallback_lark(lark_str, guidance_tokens, chat_template, tokenizer)
    {
        TopLevelGrammar::from_lark_ascii(&transformed_lark)
    } else {
        grammar
    }
}

// REASONING TOKEN FUNCTIONS

/// Extract reasoning token strings from GuidanceTokens using tokenizer
/// Returns Some((start_string, end_string)) if tokens exist, None otherwise
pub fn get_reasoning_token_strings(
    guidance_tokens: &GuidanceTokens,
    tokenizer: &Tokenizer,
) -> Option<(String, String)> {
    if guidance_tokens.reasoning_start_ids.is_empty()
        || guidance_tokens.reasoning_end_ids.is_empty()
    {
        return None;
    }

    // Use tokenizer to decode token IDs to strings
    let start_str = tokenizer
        .decode(&guidance_tokens.reasoning_start_ids, false)
        .ok()?;
    let end_str = tokenizer
        .decode(&guidance_tokens.reasoning_end_ids, false)
        .ok()?;

    Some((start_str, end_str))
}

/// Check if a TopLevelGrammar contains reasoning block patterns.
/// Looks for the `reasoning_block:` rule definition (LHS with colon) combined with
/// token ID syntax, to distinguish from user grammars that might mention "reasoning_block".
pub fn is_reasoning_grammar(grammar: &TopLevelGrammar) -> bool {
    let lark_str = get_lark_from_top_level_grammar(grammar);
    lark_str.lines().any(|l| {
        let trimmed = l.trim();
        trimmed.starts_with("reasoning_block:") && trimmed.contains("<[") && trimmed.contains("]>")
    })
}

/// Build TopLevelGrammar from a GrammarRequest
/// This function handles all grammar types (lark, regex, json_schema, choice)
/// and returns a parsed TopLevelGrammar ready for use in guided decoding.
pub fn build_grammar_from_request(
    grammar_type: &str,
    grammar_content: &str,
) -> GrammarResult<TopLevelGrammar> {
    match grammar_type {
        "lark" => Ok(TopLevelGrammar::from_lark_ascii(grammar_content)),
        "json_schema" => {
            let value: serde_json::Value = serde_json::from_str(grammar_content)
                .map_err(|e| GrammarError::InvalidGrammar(format!("Invalid JSON schema: {}", e)))?;
            let sanitized = sanitize_schema_for_llguidance(&value);
            TopLevelGrammar::from_json_schema_ascii(sanitized)
                .map_err(|e| GrammarError::InvalidGrammar(format!("Invalid schema: {}", e)))
        }
        "regex" => Ok(TopLevelGrammar::from_regex_ascii(grammar_content)),
        "choice" => {
            // Parse the grammar_content as a JSON array of strings
            let choices: Vec<String> = serde_json::from_str(grammar_content)
                .map_err(|e| GrammarError::InvalidGrammar(format!("Invalid choice JSON: {}", e)))?;
            build_choice_lark_grammar(&choices)
        }
        other => Err(GrammarError::UnsupportedFormat(format!(
            "Unknown grammar_type: {}",
            other
        ))),
    }
}

/// Build a Lark grammar for choice constraints (structured outputs choice field)
pub fn build_choice_lark_grammar(choices: &[String]) -> GrammarResult<TopLevelGrammar> {
    // Validate choices - must not contain empty strings
    for choice in choices {
        if choice.is_empty() {
            return Err(GrammarError::InvalidGrammar(
                "Choice grammar cannot contain empty strings".to_string(),
            ));
        }
    }

    // Build Lark grammar for choices using lark_quote for proper escaping
    let mut parts = Vec::with_capacity(choices.len());
    for choice in choices {
        parts.push(lark_quote(choice));
    }
    let choice_grammar = parts.join(" | ");

    // Create TopLevelGrammar from the choice Lark string
    let lark = format!("start: {}\n", choice_grammar);
    Ok(TopLevelGrammar::from_lark_ascii(&lark))
}

/// Generate complete TopLevelGrammar from ChatCompletionRequest
/// Single call-site function that handles all grammar permutations
/// Returns fully composed grammar with proper <[token_id]> format for tool tags
pub fn generate_grammar_from_request(
    request: &crate::server::ChatCompletionRequest,
    guidance_tokens: &crate::utils::guidance::GuidanceTokens,
    enable_tool_grammar: bool,
    model_type: &crate::utils::config::ModelType,
    _model_id: &str,
    parser_name: String,
    tokenizer: &Tokenizer,
    chat_template: Option<ChatTemplate>,
    disable_reasoning: bool,
) -> Option<TopLevelGrammar> {
    let tool_config = ToolConfig::from_tokenizer(tokenizer, model_type);

    let dispatcher = GrammarRequestDispatcher::new(
        request,
        guidance_tokens,
        &tool_config,
        enable_tool_grammar,
        parser_name,
        tokenizer,
        chat_template,
        disable_reasoning,
    );

    dispatcher.build_grammar()
}

/// Build guided decoding grammar for claude_server.rs.
///
/// Constructs a synthetic ChatCompletionRequest from Claude-style parameters so the
/// unified GrammarRequestDispatcher can handle grammar composition.  This adapter
/// exists because the Claude API surface differs from the OpenAI-compatible one;
/// refactoring both paths into a shared non-HTTP struct is tracked as future work.
pub fn build_guided_decoding_grammar(
    guidance_tokens: &crate::utils::guidance::GuidanceTokens,
    _tool_config: &crate::server::parser::ToolConfig,
    tools: &[crate::tools::Tool],
    tool_parser_name: &str,
    constraint_grammar: Option<TopLevelGrammar>,
    tool_choice_required: bool,
    forced_tool_name: Option<String>,
    max_tokens: usize,
    reasoning_effort: Option<crate::utils::config::ReasoningEffort>,
    enable_tool_grammar: bool,
    tokenizer: &Tokenizer,
    model_type: &crate::utils::config::ModelType,
    _model_id: &str,
    chat_template: Option<ChatTemplate>,
    disable_reasoning: bool,
) -> Option<TopLevelGrammar> {
    // If constraint_grammar is provided, extract the Lark string and set it as a constraint
    // The dispatcher will handle this in build_constraint_grammar
    let constraint = constraint_grammar
        .as_ref()
        .map(|cg| get_lark_from_top_level_grammar(cg));

    let tool_choice = forced_tool_name
        .map(crate::tools::ToolChoice::function)
        .or_else(|| tool_choice_required.then(crate::tools::ToolChoice::required));

    // Build a synthetic request with tools and constraint info
    let synthetic_request = crate::server::ChatCompletionRequest {
        messages: vec![],
        model: None,
        temperature: None,
        max_tokens: Some(max_tokens),
        top_k: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        thinking: None,
        stop: None,
        stream: None,
        stream_options: None,
        session_id: None,
        tools: if tools.is_empty() {
            None
        } else {
            Some(tools.to_vec())
        },
        tool_choice,
        response_format: None,
        extra_body: None,
        structured_outputs: None, // constraint_grammar is handled via constraint field
        constraint: constraint,
        constraint_type: Some("lark".to_string()), // constraint_grammar is always Lark format
        reasoning_effort: reasoning_effort.map(|e| e.to_string()),
    };

    // Use GrammarRequestDispatcher with disable_reasoning
    let tool_config = ToolConfig::from_tokenizer(tokenizer, model_type);
    let parser_name = tool_parser_name.to_string();

    let dispatcher = GrammarRequestDispatcher::new(
        &synthetic_request,
        guidance_tokens,
        &tool_config,
        enable_tool_grammar,
        parser_name,
        tokenizer,
        chat_template,
        disable_reasoning,
    );

    dispatcher.build_grammar()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::{ChatCompletionRequest, StructuredOutputs};
    use serde_json::json;
    use tokenizers::{models::bpe::BPE, AddedToken, Tokenizer};

    fn guidance_tokens() -> GuidanceTokens {
        GuidanceTokens {
            bos_token_ids: vec![151647],
            eos_token_ids: vec![151648],
            reasoning_start_ids: vec![151657],
            reasoning_end_ids: vec![151658],
            tool_call_start_ids: vec![151657],
            tool_call_end_ids: vec![151658],
            add_bos_token: false,
        }
    }

    fn request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            messages: vec![],
            model: None,
            temperature: None,
            max_tokens: Some(16),
            top_k: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            thinking: None,
            stop: None,
            stream: None,
            stream_options: None,
            session_id: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            extra_body: None,
            structured_outputs: None,
            constraint: None,
            constraint_type: None,
            reasoning_effort: None,
        }
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::new(BPE::default())
    }

    fn glm_marker_tokenizer() -> Tokenizer {
        let mut tokenizer = tokenizer();
        tokenizer.add_tokens(&[
            AddedToken::from("<arg_key>", false),
            AddedToken::from("</arg_key>", false),
            AddedToken::from("<arg_value>", false),
            AddedToken::from("</arg_value>", false),
        ]);
        tokenizer
    }

    #[test]
    fn test_build_choice_lark_grammar_rejects_empty_choice() {
        let result = build_choice_lark_grammar(&["".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_lark_quote_escapes_special_chars() {
        let result = lark_quote("test\"value");
        assert!(result.contains("test\\\"value"));
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_strips_metadata_and_resolves_refs() {
        let schema = json!({
            "$defs": {
                "Item": {
                    "description": "metadata",
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "default": "x", "description": "metadata"},
                        "count": {"type": "integer", "minimum": 1, "maximum": 3}
                    },
                    "required": ["name"]
                }
            },
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
            },
            "title": "metadata"
        });

        let sanitized = sanitize_schema_for_llguidance(&schema);

        assert!(sanitized.get("$defs").is_none());
        assert!(sanitized.get("title").is_none());
        let item = &sanitized["properties"]["items"]["items"];
        assert!(item.get("$ref").is_none());
        assert_eq!(item["type"], "object");
        assert!(item["properties"]["name"].get("default").is_none());
        assert!(item["properties"]["name"].get("description").is_none());
        assert_eq!(item["properties"]["count"]["minimum"], 1);
    }

    #[test]
    fn test_structured_outputs_choice_dispatch_builds_grammar() {
        let mut req = request();
        req.structured_outputs = Some(StructuredOutputs {
            choice: Some(vec!["yes".to_string(), "no".to_string()]),
            regex: None,
            json: None,
            grammar: None,
            structural_tag: None,
        });

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            true,
            "json".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar()
        .expect("choice grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("yes"));
        assert!(lark.contains("no"));
        assert!(lark.contains("eos"));
    }

    #[test]
    fn test_structured_outputs_builds_grammar_when_tool_grammar_disabled() {
        let mut req = request();
        req.structured_outputs = Some(StructuredOutputs {
            choice: Some(vec!["yes".to_string(), "no".to_string()]),
            regex: None,
            json: None,
            grammar: None,
            structural_tag: None,
        });

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            false,
            "json".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar()
        .expect("structured output grammar should be built without tool grammar");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("yes"));
        assert!(lark.contains("no"));
    }

    #[test]
    fn test_tools_do_not_build_grammar_when_tool_grammar_disabled() {
        let mut req = request();
        req.tools = Some(vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()]);

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            false,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar();

        assert!(grammar.is_none());
    }

    #[test]
    fn test_qwen_coder_tool_dispatch_builds_xml_grammar() {
        let mut req = request();
        req.tools = Some(vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()]);

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            true,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar()
        .expect("tool grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("start:"));
        assert!(lark.contains("text"));
        assert!(lark.contains("tool_call"));
        assert!(lark.contains("<function=search>"));
        assert!(lark.contains("parameter=query"));
    }

    #[test]
    fn test_auto_tool_grammar_allows_text_response() {
        let mut req = request();
        req.tool_choice = Some(crate::tools::ToolChoice::auto());
        req.tools = Some(vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()]);

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            true,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar()
        .expect("auto tool grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("text"));
        assert!(lark.contains("tool_call"));
        assert!(!lark.contains("start: tool_call\n"));
    }

    #[test]
    fn test_required_tool_grammar_forces_tool_call() {
        let mut req = request();
        req.tool_choice = Some(crate::tools::ToolChoice::required());
        req.tools = Some(vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()]);

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            true,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar()
        .expect("required tool grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("start: tool_call"));
    }

    #[test]
    fn test_claude_auto_tool_grammar_allows_text_response() {
        let tools = vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()];

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = build_guided_decoding_grammar(
            &tokens,
            &tool_config,
            &tools,
            "qwen_coder",
            None,
            false,
            None,
            16,
            None,
            true,
            &tokenizer,
            &crate::utils::config::ModelType::Qwen3,
            "qwen",
            None,
            false,
        )
        .expect("Claude auto tool grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("text"));
        assert!(lark.contains("tool_call"));
        assert!(!lark.contains("start: tool_call\n"));
    }

    #[test]
    fn test_claude_required_tool_grammar_forces_tool_call() {
        let tools = vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()];

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = build_guided_decoding_grammar(
            &tokens,
            &tool_config,
            &tools,
            "qwen_coder",
            None,
            true,
            None,
            16,
            None,
            true,
            &tokenizer,
            &crate::utils::config::ModelType::Qwen3,
            "qwen",
            None,
            false,
        )
        .expect("Claude required tool grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("start: tool_call"));
    }

    #[test]
    fn test_tool_choice_none_does_not_build_tool_grammar() {
        let mut req = request();
        req.tool_choice = Some(crate::tools::ToolChoice::none());
        req.tools = Some(vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()]);

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            true,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar();

        assert!(grammar.is_none());
    }

    #[test]
    fn test_tool_grammar_does_not_append_duplicate_tool_end_as_eos() {
        let mut req = request();
        req.tools = Some(vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()]);

        let mut tokens = guidance_tokens();
        tokens.eos_token_ids = tokens.tool_call_end_ids.clone();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            true,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar()
        .expect("tool grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("tool_call"));
        assert!(!lark.contains("tool_call eos"));
        assert!(!lark.contains("eos:"));
    }

    #[test]
    fn test_tool_grammar_uses_parser_tool_config_token_ids() {
        let mut req = request();
        req.tools = Some(vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()]);

        let mut tokens = guidance_tokens();
        tokens.tool_call_start_ids = vec![42];
        tokens.tool_call_end_ids = vec![43];
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::Qwen3);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            true,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar()
        .expect("tool grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("<[151657]>"));
        assert!(lark.contains("<[151658]>"));
        assert!(!lark.contains("<[42]>"));
        assert!(!lark.contains("<[43]>"));
    }

    #[test]
    fn test_glm_tool_dispatch_builds_native_xml_grammar() {
        let mut req = request();
        req.tools = Some(vec![crate::tools::ToolBuilder::new(
            "read".to_string(),
            "Read a file".to_string(),
        )
        .param("filePath", "string", "File path", true)
        .build()]);

        let tokens = guidance_tokens();
        let tokenizer = tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::GLM4MoeLite);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            true,
            "glm47_moe".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar()
        .expect("GLM tool grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("tool_call"));
        assert!(lark.contains("read"));
        assert!(lark.contains("glm_arg_pair"));
        assert!(lark.contains("glm_arg_key"));
        assert!(!lark.contains("glm_arg_key?"));
        assert!(lark.contains("glm_arg_value?"));
        assert!(lark.contains(r#"glm_arg_key: "filePath""#));
        assert!(lark.contains(r#"glm_arg_value[suffix="</arg_value>"]: /(?s:.+?)/"#));
        assert!(lark.contains("<arg_key>"));
        assert!(lark.contains("</arg_key>"));
        assert!(lark.contains("<arg_value>"));
        assert!(lark.contains("</arg_value>"));
        assert!(!lark.contains(r#""name""#));
        assert!(!lark.contains(r#""arguments""#));
    }

    #[test]
    fn test_glm_tool_dispatch_uses_nested_marker_token_ids() {
        let mut req = request();
        req.tools = Some(vec![crate::tools::ToolBuilder::new(
            "list_dir".to_string(),
            "List directory contents".to_string(),
        )
        .param("path", "string", "Directory path", true)
        .build()]);

        let tokens = guidance_tokens();
        let tokenizer = glm_marker_tokenizer();
        let tool_config = ToolConfig::for_model_type(&crate::utils::config::ModelType::GLM4MoeLite);
        let grammar = GrammarRequestDispatcher::new(
            &req,
            &tokens,
            &tool_config,
            true,
            "glm47_moe".to_string(),
            &tokenizer,
            None,
            false,
        )
        .build_grammar()
        .expect("GLM tool grammar should be built");

        let lark = get_lark_from_top_level_grammar(&grammar);
        for marker in ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"] {
            let encoded = tokenizer.encode(marker, false).unwrap();
            let ids = encoded.get_ids();
            assert_eq!(ids.len(), 1);
            assert!(
                lark.contains(&format!("<[{}]>", ids[0])),
                "grammar should contain token terminal for {marker}"
            );
        }
        assert!(!lark.contains("</arg_key><arg_value>"));
        assert!(lark.contains(r#"glm_arg_key: "path""#));
        let end_value_id = tokenizer.encode("</arg_value>", false).unwrap().get_ids()[0];
        assert!(lark.contains(&format!("glm_arg_value: <[^{},151658]>+", end_value_id)));
        assert!(!lark.contains(r#"suffix="</arg_value>""#));
    }

    #[test]
    fn test_build_grammar_from_request_json_schema_sanitizes() {
        let grammar = build_grammar_from_request(
            "json_schema",
            r#"{"type":"object","properties":{"name":{"type":"string","description":"metadata"}},"required":["name"]}"#,
        )
        .expect("json schema grammar should compile");

        let lark = get_lark_from_top_level_grammar(&grammar);
        assert!(lark.contains("%json"));
        assert!(lark.contains("name"));
        assert!(!lark.contains("metadata"));
    }
}
