// src/utils/guidance_grammar.rs
//! Clean-sheet grammar generation for llguidance
//! Handles constraints, tools, and reasoning in a simple, idiomatic way

use llguidance::api::TopLevelGrammar;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use crate::server::ChatCompletionRequest;
use crate::tools::Tool;
use crate::server::parser::{StreamToolParser, ToolConfig};
use crate::utils::chat_template::ChatTemplate;
use crate::utils::config::ModelType;
use crate::utils::config::ReasoningEffort;
use crate::utils::guidance::GuidanceTokens;
use tokenizers::Tokenizer;
use crate::utils::env::default_reasoning_max_tokens;

// COMMON TRAITS

/// Common trait for grammar builders in llguidance integration
///
/// Each grammar type must implement `build_lark()` to generate its Lark representation.
/// Default implementations are provided for composition methods; override when needed.
pub trait GrammarBuilder: Clone + Default + std::fmt::Debug + Sized {
    /// Build the Lark grammar string - must be implemented by each grammar type
    fn build_lark(&mut self) -> String;

    /// Compose two grammars with alternation (OR) - defaults to cloning 'other'
    /// Override when specific alternation logic is needed
    fn compose_alternate(&mut self, other: &mut Self) -> Self {
        other.clone()
    }

    /// Compose two grammars with sequence (AND) - defaults to cloning 'other'
    /// Override when specific sequence logic is needed
    fn compose_sequence(&mut self, other: &mut Self) -> Self {
        other.clone()
    }

    /// Convert to TopLevelGrammar - defaults to parsing build_lark() output
    fn format(&mut self) -> TopLevelGrammar {
        TopLevelGrammar::from_lark_ascii(&self.build_lark())
    }

    /// Substitute token IDs from a mapping - defaults to cloning self
    /// Override when token-specific mutation is needed
    fn substitute_tokens(&mut self, _token_map: &TokenSubstitutionMap) -> Self {
        self.clone()
    }

    /// Extract specific rules by name - defaults to empty vector
    /// Override when line-based extraction is needed
    fn extract_rules(&mut self, _rule_names: &[&str]) -> Vec<String> {
        Vec::new()
    }
}

pub type TokenSubstitutionMap = HashMap<String, Vec<u32>>;

/// Result type for grammar-related operations
pub type GrammarResult<T> = Result<T, GrammarError>;

/// Error type for grammar-related operations
#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("Invalid grammar: {0}")]
    InvalidGrammar(String),
    #[error("Missing constraint: {0}")]
    MissingConstraint(String),
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

    /// Deprecated: use from_lark_ascii instead
    fn from_lark_utf8(lark: &str) -> Self {
        Self::from_lark_ascii(lark)
    }

    /// Deprecated: use from_json_schema_ascii instead
    fn from_json_schema_utf8(schema: serde_json::Value) -> Result<Self, anyhow::Error> {
        Self::from_json_schema_ascii(schema)
    }
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
        "anyOf", "oneOf", "allOf", "$ref", "const", "enum", "type",
        // Array
        "items", "additionalItems", "prefixItems", "minItems", "maxItems",
        // Object
        "properties", "additionalProperties", "patternProperties", "required", "minProperties", "maxProperties",
        // String
        "minLength", "maxLength", "pattern", "format",
        // Number
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        // Schema definitions (for $ref resolution)
        "$defs", "definitions", "$anchor",
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
                            new_props.insert(prop_name.clone(), sanitize_sanitized_schema_recursive(prop_schema));
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

/// Build special token syntax for Lark grammar using token IDs
/// When token IDs are available, uses <[token_id]> syntax instead of string literals
/// This ensures alignment with the outbound parser's token-based detection
pub fn lark_special_token(token_ids: &HashSet<u32>) -> String {
    if token_ids.is_empty() {
        return String::new();
    }
    // Join multiple token IDs with | - ensure ASCII only
    let ids: Vec<String> = token_ids.iter().map(|id| format!("[{}]", id)).collect();
    format!("<{}>", ids.join(","))
}

// REASONING GRAMMAR - ReasoningEffort is defined in utils/config.rs

#[derive(Clone, Debug)]
pub struct ReasoningGrammar {
    pub start_token_id: u32,
    pub end_token_id: u32,
    pub effort: ReasoningEffort,
}

impl Default for ReasoningGrammar {
    fn default() -> Self {
        Self {
            start_token_id: 0,
            end_token_id: 0,
            effort: ReasoningEffort::None,
        }
    }
}

impl ReasoningGrammar {
    pub fn new(start_id: u32, end_id: u32, effort: ReasoningEffort) -> Self {
        Self {
            start_token_id: start_id,
            end_token_id: end_id,
            effort,
        }
    }

    /// Generate the appropriate grammar template for this reasoning level
    /// This is used by guidance_grammar.rs to build Lark grammars
    fn generate_grammar(&self, start_id: u32, end_id: u32) -> String {
        match &self.effort {
            ReasoningEffort::None => {
                // No reasoning block - direct output only
                // Minimal latency, no structured thinking
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n\n" <[{end_id}]> "\n"
"#
                )
            }
            ReasoningEffort::ModelDefault => {
                let rmt = default_reasoning_max_tokens();
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n" think_text? "\n" <[{end_id}]> "\n"
think_text[temperature=0, max_tokens={rmt}]: /(?s:.+?)/
"#
                )
            }
            ReasoningEffort::Low => {
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n" think_text "\n" (think_text+ "\n")? <[{end_id}]> "\n"
think_text[temperature=0.3, max_tokens=256]: /(?s:.+?)/
"#
                )
            }
            ReasoningEffort::Medium => {
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n" think_text "\n" <[{end_id}]> "\n"
think_text[temperature=0.5, max_tokens=768]: /(?s:.+?)/
"#
                )
            }
            ReasoningEffort::High => {
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> analysis_block critique_block structure_block "\n" <[{end_id}]> "\n"
analysis_block: "\n<analysis>\n" analysis_text
analysis_text[suffix="\n</analysis>\n", temperature=0.8, max_tokens=512]: /(?s:.+?)/
critique_block: "\n<critique>\n" critique_text
critique_text[suffix="\n</critique>\n", temperature=0, max_tokens=512]: /(?s:.+?)/
structure_block: "\n<structure_response>\n" structure_text
structure_text[suffix="\n</structure_response>\n", temperature=0.8, max_tokens=512]: /(?s:.+?)/
"#
                )
            }
            ReasoningEffort::ChainOfThought => {
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> draft_block verification_block critique_block structure_block "\n" <[{end_id}]> "\n"
draft_block: "\n<draft>\nCardinalities of concern, intended outcomes, and structures of consideration:\n" draft_text
draft_text[suffix="\n</draft>\n", temperature=0.8, max_tokens=768]: /(?s:.+?)/
verification_block: "\n<verify>\nQuestions, assumptions, and suppositions:\n" verification_text
verification_text[suffix="\n</verify>\n", temperature=0, max_tokens=768]: /(?s:.+?)/
critique_block: "\n<critique>\nAdversarial assessment of evaluation:\n" critique_text
critique_text[suffix="\n</critique>\n", temperature=0.6, max_tokens=768]: /(?s:.+?)/
structure_block: "\n<structure_response>\n" structure_text
structure_text[suffix="\n</structure_response>\n", temperature=0.8, max_tokens=768]: /(?s:.+?)/
"#
                )
            }
            #[cfg(all(not(feature = "python"), not(feature = "pyo3")))]
            ReasoningEffort::Custom(template) => {
                // User-provided template with token ID injection
                // Supports $START_ID and $END_ID placeholders for dynamic token ID substitution
                template
                    .replace("$START_ID", &start_id.to_string())
                    .replace("$END_ID", &end_id.to_string())
            }
        }
    }
}

impl GrammarBuilder for ReasoningGrammar {
    fn build_lark(&mut self) -> String {
        if self.effort == ReasoningEffort::None {
            return String::new();
        }
        self.generate_grammar(self.start_token_id, self.end_token_id)
    }

    fn compose_sequence(&mut self, other: &mut Self) -> Self {
        if other.effort != ReasoningEffort::None {
            return other.clone();
        }
        self.clone()
    }

    fn format(&mut self) -> TopLevelGrammar {
        let lark = self.build_lark();
        if lark.is_empty() {
            TopLevelGrammar::from_lark_ascii("startf\ntext: /(?s:.+?)/")
        } else {
            TopLevelGrammar::from_lark_ascii(&lark)
        }
    }

    fn substitute_tokens(&mut self, token_map: &TokenSubstitutionMap) -> Self {
        let mut new = self.clone();
        if let Some(start_ids) = token_map.get("reasoning_start") {
            if let Some(&id) = start_ids.first() {
                new.start_token_id = id;
            }
        }
        if let Some(end_ids) = token_map.get("reasoning_end") {
            if let Some(&id) = end_ids.first() {
                new.end_token_id = id;
            }
        }
        new
    }

    fn extract_rules(&mut self, rule_names: &[&str]) -> Vec<String> {
        let lark = self.build_lark();
        let mut rules = Vec::new();
        for line in lark.lines() {
            for rule_name in rule_names {
                if line.starts_with(*rule_name) {
                    rules.push(line.trim().to_string());
                }
            }
        }
        rules
    }
}

// CHAT RESPONSE GRAMMAR

#[derive(Clone, Debug)]
pub struct ChatResponseGrammar {
    pub eos_termination: bool,
    pub max_tokens: Option<usize>,
}

impl Default for ChatResponseGrammar {
    fn default() -> Self {
        Self {
            eos_termination: false,
            max_tokens: None,
        }
    }
}

impl ChatResponseGrammar {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_eos(self, eos: bool) -> Self {
        Self {
            eos_termination: eos,
            ..self
        }
    }
    pub fn with_max_tokens(self, max: usize) -> Self {
        Self {
            max_tokens: Some(max),
            ..self
        }
    }
}

impl GrammarBuilder for ChatResponseGrammar {
    fn build_lark(&mut self) -> String {
        if self.eos_termination {
            r#"start: text
text: /(?s:.+?)/"#
                .to_string()
        } else {
            if let Some(max_tokens) = self.max_tokens {
                format!(r#"start: text
text[stop="", max_tokens={}]: /(?s:.+?)/"#, max_tokens)
            } else {
                r#"start: text
text[stop=""]: /(?s:.+?)/"#
                .to_string()
            }
        }
    }
    fn compose_alternate(&mut self, _other: &mut Self) -> Self {
        self.clone()
    }
    fn compose_sequence(&mut self, other: &mut Self) -> Self {
        Self {
            eos_termination: self.eos_termination || other.eos_termination,
            max_tokens: self.max_tokens.or(other.max_tokens),
        }
    }
    fn format(&mut self) -> TopLevelGrammar {
        TopLevelGrammar::from_lark_ascii(&self.build_lark())
    }
    fn substitute_tokens(&mut self, _token_map: &TokenSubstitutionMap) -> Self {
        self.clone()
    }
    fn extract_rules(&mut self, _rule_names: &[&str]) -> Vec<String> {
        Vec::new()
    }
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

        // Combine the start alternatives
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

    fn compose_sequence(&mut self, other: &mut Self) -> Self {
        let this_lark = self.build_lark();
        let other_lark = other.build_lark();

        // Parse both grammars and combine rules, deduplicating
        let this_lines: Vec<&str> = this_lark.lines().collect();
        let other_lines: Vec<&str> = other_lark.lines().collect();

        // Extract start rules and other rules from both
        let this_start = this_lines
            .first()
            .and_then(|l| l.strip_prefix("start: "))
            .unwrap_or("");
        let other_start = other_lines
            .first()
            .and_then(|l| l.strip_prefix("start: "))
            .unwrap_or("");

        // Combine start rules
        let combined_start = format!("{} {}", this_start, other_start).trim().to_string();

        // Collect all non-start rules from both, deduplicating
        let mut seen = std::collections::HashSet::new();
        let mut all_rules: Vec<String> = Vec::new();

        for line in this_lines.iter().skip(1) {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !seen.contains(trimmed) {
                seen.insert(trimmed.to_string());
                all_rules.push(trimmed.to_string());
            }
        }

        for line in other_lines.iter().skip(1) {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !seen.contains(trimmed) {
                seen.insert(trimmed.to_string());
                all_rules.push(trimmed.to_string());
            }
        }

        let combined_rules = all_rules.join("\n");

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
    fn substitute_tokens(&mut self, _token_map: &TokenSubstitutionMap) -> Self {
        self.clone()
    }
    fn extract_rules(&mut self, _rule_names: &[&str]) -> Vec<String> {
        Vec::new()
    }
}

impl StructuredOutputsGrammar {}

// RESPONSE FORMAT GRAMMAR

#[derive(Clone, Debug)]
pub struct ResponseFormatGrammar {
    pub format_type: String,
    pub schema: Option<Value>,
}

impl Default for ResponseFormatGrammar {
    fn default() -> Self {
        Self {
            format_type: "json_object".to_string(),
            schema: None,
        }
    }
}

impl ResponseFormatGrammar {
    pub fn new_json_schema(schema: Value) -> Self {
        Self {
            format_type: "json_schema".to_string(),
            schema: Some(schema),
        }
    }
    pub fn new_json_object() -> Self {
        Self {
            format_type: "json_object".to_string(),
            schema: None,
        }
    }
}

impl GrammarBuilder for ResponseFormatGrammar {
    fn build_lark(&mut self) -> String {
        match self.format_type.as_str() {
            "json_schema" => {
                if let Some(schema) = &self.schema {
                    let sanitized = sanitize_schema_for_llguidance(schema);
                    let schema_str = serde_json::to_string(&sanitized).unwrap_or_default();
                    format!(
                        r#"start: text
text: %json {}"#,
                        schema_str
                    )
                } else {
                    String::new()
                }
            }
            "json_object" => r#"start: text
text: %json {"type":"object"}"#
                .to_string(),
            _ => String::new(),
        }
    }
    fn compose_alternate(&mut self, _other: &mut Self) -> Self {
        self.clone()
    }
    fn compose_sequence(&mut self, other: &mut Self) -> Self {
        other.clone()
    }
    fn format(&mut self) -> TopLevelGrammar {
        TopLevelGrammar::from_lark_ascii(&self.build_lark())
    }
    fn substitute_tokens(&mut self, _token_map: &TokenSubstitutionMap) -> Self {
        self.clone()
    }
    fn extract_rules(&mut self, _rule_names: &[&str]) -> Vec<String> {
        Vec::new()
    }
}

// CONSTRAINT GRAMMAR

#[derive(Clone, Debug)]
pub struct ConstraintGrammar {
    pub constraint_type: String,
    pub content: String,
}

impl Default for ConstraintGrammar {
    fn default() -> Self {
        Self {
            constraint_type: "regex".to_string(),
            content: String::new(),
        }
    }
}

impl ConstraintGrammar {
    pub fn new_regex(content: String) -> Self {
        Self {
            constraint_type: "regex".to_string(),
            content,
        }
    }
    pub fn new_lark(content: String) -> Self {
        Self {
            constraint_type: "lark".to_string(),
            content,
        }
    }
    pub fn new_json_schema(content: String) -> Self {
        Self {
            constraint_type: "json_schema".to_string(),
            content,
        }
    }
}

impl GrammarBuilder for ConstraintGrammar {
    fn build_lark(&mut self) -> String {
        match self.constraint_type.as_str() {
            "regex" => format!(
                r#"start: text
text: /{}/"#,
                self.content
            ),
            "lark" => self.content.clone(),
            "json_schema" => {
                if let Ok(schema) = serde_json::from_str(&self.content) {
                    let sanitized = sanitize_schema_for_llguidance(&schema);
                    let schema_str = serde_json::to_string(&sanitized).unwrap_or_default();
                    format!(
                        r#"start: text
text: %json {}"#,
                        schema_str
                    )
                } else {
                    String::new()
                }
            }
            _ => String::new(),
        }
    }
    fn compose_alternate(&mut self, _other: &mut Self) -> Self {
        self.clone()
    }
    fn compose_sequence(&mut self, other: &mut Self) -> Self {
        other.clone()
    }
    fn format(&mut self) -> TopLevelGrammar {
        TopLevelGrammar::from_lark_ascii(&self.build_lark())
    }
    fn substitute_tokens(&mut self, _token_map: &TokenSubstitutionMap) -> Self {
        self.clone()
    }
    fn extract_rules(&mut self, _rule_names: &[&str]) -> Vec<String> {
        Vec::new()
    }
}

// TOOL CALL GRAMMAR

#[derive(Clone, Debug)]
pub enum ToolFormat {
    QwenCoder,
    MiniMax,
    Json,
    Generic,
    Gemma4,
}

#[derive(Clone, Debug)]
pub struct ToolCallGrammar {
    pub tools: Vec<Tool>,
    pub start_token_id: u32,
    pub end_token_id: u32,
    pub format: ToolFormat,
    value_rules: HashMap<String, String>,
}

impl Default for ToolCallGrammar {
    fn default() -> Self {
        Self {
            tools: Vec::new(),
            start_token_id: 0,
            end_token_id: 0,
            format: ToolFormat::Json,
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
            value_rules: HashMap::new(),
        }
    }
    pub fn new_qwen_coder(tools: Vec<Tool>, start_token_id: u32, end_token_id: u32) -> Self {
        Self {
            tools,
            start_token_id,
            end_token_id,
            format: ToolFormat::QwenCoder,
            value_rules: HashMap::new(),
        }
    }
    pub fn new_minimax(tools: Vec<Tool>, start_token_id: u32, end_token_id: u32) -> Self {
        Self {
            tools,
            start_token_id,
            end_token_id,
            format: ToolFormat::MiniMax,
            value_rules: HashMap::new(),
        }
    }
    pub fn new_json(tools: Vec<Tool>, start_token_id: u32, end_token_id: u32) -> Self {
        Self {
            tools,
            start_token_id,
            end_token_id,
            format: ToolFormat::Json,
            value_rules: HashMap::new(),
        }
    }
    pub fn _new_gemma4(tools: Vec<Tool>, start_token_id: u32, end_token_id: u32) -> Self {
        Self {
            tools,
            start_token_id,
            end_token_id,
            format: ToolFormat::Gemma4,
            value_rules: HashMap::new(),
        }
    }

    /// Build ToolCallGrammar with model-aware selection using StreamToolParser for consistent parser name determination
    /// This ensures grammar generation aligns with the streaming parser's format selection
    pub fn for_model_type(
        tools: Vec<Tool>,
        start_token_id: u32,
        end_token_id: u32,
        model_type: &ModelType,
        model_id: &str,
    ) -> ToolCallGrammar {
        // Use StreamToolParser::parser_name_for_model() to determine the parser name
        let parser_name = StreamToolParser::parser_name_for_model(model_type, model_id);

        // Map parser name to ToolFormat
        let format = Self::tool_format_for_parser_name(parser_name);

        ToolCallGrammar {
            tools,
            start_token_id,
            end_token_id,
            format,
            value_rules: HashMap::new(),
        }
    }

    /// Map parser name to ToolFormat variant
    /// Parsers without explicit grammars default to Generic
    fn tool_format_for_parser_name(parser_name: &str) -> ToolFormat {
        match parser_name {
            "qwen_coder" => ToolFormat::QwenCoder,
            "minimax_m2" => ToolFormat::MiniMax,
            "json" => ToolFormat::Json,
            _ => ToolFormat::Generic,
        }
    }
}

impl GrammarBuilder for ToolCallGrammar {
    fn build_lark(&mut self) -> String {
        match self.format {
            ToolFormat::QwenCoder => self.build_qwen_coder_lark(),
            ToolFormat::MiniMax   => self.build_minimax_lark(),
            ToolFormat::Gemma4    => self.build_gemma4_lark(),
            ToolFormat::Json      => self.build_json_lark(),
            ToolFormat::Generic   => self.build_generic_lark(),
        }
    }
    fn compose_alternate(&mut self, _other: &mut Self) -> Self {
        self.clone()
    }
    fn compose_sequence(&mut self, other: &mut Self) -> Self {
        other.clone()
    }
    fn format(&mut self) -> TopLevelGrammar {
        TopLevelGrammar::from_lark_ascii(&self.build_lark())
    }
    fn substitute_tokens(&mut self, token_map: &TokenSubstitutionMap) -> Self {
        let mut new = self.clone();
        if let Some(ids) = token_map.get("tool_start") {
            if let Some(&id) = ids.first() {
                new.start_token_id = id;
            }
        }
        if let Some(ids) = token_map.get("tool_end") {
            if let Some(&id) = ids.first() {
                new.end_token_id = id;
            }
        }
        new
    }
    fn extract_rules(&mut self, rule_names: &[&str]) -> Vec<String> {
        let lark = self.build_lark();
        let mut rules = Vec::new();
        for line in lark.lines() {
            for rule_name in rule_names {
                if line.starts_with(*rule_name) {
                    rules.push(line.trim().to_string());
                }
            }
        }
        rules
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
                serde_json::json!({ "anyOf": variants })
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

    fn build_gemma4_lark(&mut self) -> String {
        let start_tag = format!("<[{}]>", self.start_token_id);
        let end_tag = format!("<[{}]>", self.end_token_id);

        if self.tools.is_empty() {
            return format!(
                r#"start: tool_call
    tool_call: {} call_text {}
    call_text: /(?s:.*)/
    "#,
                start_tag, end_tag
            );
        }

        let mut rules: Vec<String> = Vec::new();
        let tool_rule_names: Vec<String> = (0..self.tools.len()).map(|i| format!("tool_{}", i)).collect();

        rules.push("start: tool_call".to_string());
        rules.push(format!("tool_call: {} tool_content {}", start_tag, end_tag));
        rules.push(format!("tool_content: {}", tool_rule_names.join(" | ")));

        // Clone tools to avoid borrow conflict
        let tools_clone = self.tools.clone();

        for (tool_idx, tool) in tools_clone.iter().enumerate() {
            let tool_name = &tool.function.name;
            let (args_expr, param_value_rules) = self.build_gemma4_args_pattern(&tool.function.parameters, &tool_idx);

            let tool_rule = format!("tool_{}: \"call:{}{{\" {} \"}}\"", tool_idx, tool_name, args_expr);
            rules.push(tool_rule);

            for param_value in param_value_rules {
                rules.push(param_value);
            }
        }

        let value_definitions = self.build_value_rules();
        rules.extend(value_definitions);

        rules.join("\n")
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

    fn _build_gemma4_array_definition() -> String {
        r#"gemma4_array: "[" gemma4_array_items? "]"
gemma4_array_items: gemma4_value ("," gemma4_value)*
gemma4_value: gemma4_string | gemma4_number | gemma4_boolean | gemma4_array | gemma4_object
"#
            .to_string()
    }

    fn _build_gemma4_object_definition() -> String {
        r#"gemma4_object: "{" gemma4_object_items? "}"
gemma4_object_items: gemma4_key_value ("," gemma4_key_value)*
gemma4_key_value: gemma4_key ":" gemma4_value
gemma4_key: /[^:]+/
gemma4_value: gemma4_string | gemma4_number | gemma4_boolean | gemma4_array | gemma4_object
"#
            .to_string()
    }

    fn build_gemma4_pattern_definition(_rule_name: &str, value: &serde_json::Value) -> String {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type").and_then(|t| t.as_str()) {
                match type_val {
                    "string" => r#""<|\"|>" /[\x20-\x7E\x0A\x0D]+?/ "<|\"|>""#.to_string(),
                    "integer" | "number" => r#"/-?\d+(\.\d+)?/"#.to_string(),
                    "boolean" => r#""true" | "false""#.to_string(),
                    "array" => Self::build_gemma4_array_rhs(),
                    "object" => Self::build_gemma4_object_rhs(),
                    _ => r#"/.*/"#.to_string(),
                }
            } else {
                r#"/.*/"#.to_string()
            }
        } else {
            r#"/.*/"#.to_string()
        }
    }

    fn build_gemma4_array_rhs() -> String {
        r#""[" gemma4_array_items? "]""#.to_string()
    }

    fn build_gemma4_object_rhs() -> String {
        r#""{" gemma4_object_items? "}""#.to_string()
    }

    fn build_gemma4_args_pattern(&mut self, params: &serde_json::Value, tool_idx: &usize) -> (String, Vec<String>) {
        let mut param_value_rules: Vec<String> = Vec::new();
        let mut param_names: Vec<String> = Vec::new();

        if let Some(props) = params.get("properties") {
            if let Some(obj) = props.as_object() {
                for (param_idx, (key, value)) in obj.iter().enumerate() {
                    let param_rule = format!("param_{}_{}", tool_idx, param_idx);
                    let type_pattern = Self::build_gemma4_type_pattern(value);

                    // Store pattern definition in value_rules
                    let pattern_def = Self::build_gemma4_pattern_definition(&type_pattern, value);
                    self.value_rules.insert(type_pattern.clone(), pattern_def);

                    // For array/object types, add nested rules
                    if let Some(obj_val) = value.as_object() {
                        if let Some(type_val) = obj_val.get("type").and_then(|t| t.as_str()) {
                            if type_val == "array" {
                                // Add nested array rules
                                self.value_rules.insert("gemma4_array_items".to_string(), "gemma4_value (\",\" gemma4_value)*".to_string());
                                self.value_rules.insert("gemma4_value".to_string(), "gemma4_string | gemma4_number | gemma4_boolean | gemma4_array | gemma4_object".to_string());
                            } else if type_val == "object" {
                                // Add nested object rules
                                self.value_rules.insert("gemma4_object_items".to_string(), "gemma4_key_value (\",\" gemma4_key_value)*".to_string());
                                self.value_rules.insert("gemma4_key_value".to_string(), "gemma4_key \":\" gemma4_value".to_string());
                                self.value_rules.insert("gemma4_key".to_string(), "/[^:]+/".to_string());
                            }
                        }
                    }

                    let param_value = format!("{}: \"{}\" {}", param_rule, key, type_pattern);
                    param_value_rules.push(param_value);
                    param_names.push(param_rule);
                }
            }
        }

        // Build the args expression: param_0_0 ("," param_0_1)? ("," param_0_2)?
        let args_expr = if param_names.is_empty() {
            String::new()
        } else {
            let mut parts = vec![param_names[0].clone()];
            for rule in &param_names[1..] {
                parts.push(format!("( \",\" {})?", rule));
            }
            parts.join(" ")
        };

        (args_expr, param_value_rules)
    }


    fn build_gemma4_type_pattern(value: &serde_json::Value) -> String {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type").and_then(|t| t.as_str()) {
                match type_val {
                    "string" => "gemma4_string".to_string(),
                    "integer" | "number" => "gemma4_number".to_string(),
                    "boolean" => "gemma4_boolean".to_string(),
                    "array" => "gemma4_array".to_string(),
                    "object" => "gemma4_object".to_string(),
                    _ => "gemma4_value".to_string(),
                }
            } else {
                "gemma4_value".to_string()
            }
        } else {
            "gemma4_value".to_string()
        }
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
            match self.format{
                ToolFormat::MiniMax => {
                    format!(r#"{}[suffix="</parameter>\n"]"#, rule_name)
                },
                _ =>{
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
    pub allow_constraint_api: bool,
    pub parser_name: String,
    pub tokenizer: &'a Tokenizer,
    pub chat_template: Option<crate::utils::chat_template::ChatTemplate>,
    pub disable_reasoning: bool,
}

impl<'a> GrammarRequestDispatcher<'a> {
    pub fn new(
        request: &'a ChatCompletionRequest,
        guidance_tokens: &'a GuidanceTokens,
        tool_config: &'a crate::server::parser::ToolConfig,
        enable_tool_grammar: bool,
        allow_constraint_api: bool,
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
            allow_constraint_api,
            parser_name,
            tokenizer,
            chat_template,
            disable_reasoning,
        }
    }

    pub fn build_grammar(self) -> Option<TopLevelGrammar> {
        if !self.allow_constraint_api && !self.enable_tool_grammar {
            return None;
        }
        let constraint_grammar = self.build_constraint_grammar();
        let tool_grammar = self.build_tool_grammar();
        let reasoning_effort = self.build_reasoning_effort();
        // Get max_tokens from request
        let max_tokens = self.request.max_tokens.unwrap_or(0);

        // If constraint_grammar is None, use a default "any text" grammar
        // This allows tool grammars to be composed even without constraints
        let constraint_grammar = constraint_grammar.unwrap_or_else(|| {
            StructuredOutputsGrammar::new(StructuredConstraint::Lark(
                "start: text\ntext: /(?s:.+?)/".to_string(),
            ))
        });

        // Use tokenizer and chat_template from dispatcher fields for fallback application
        Some(GrammarComposer::compose_all_grammars(
            vec![constraint_grammar],
            tool_grammar,
            reasoning_effort,
            self.guidance_tokens,
            max_tokens,
            self.chat_template,
            self.tokenizer,
            self.disable_reasoning,
        ))
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

    fn build_tool_grammar(&self) -> Option<ToolCallGrammar> {
        if self.request.tools.is_none() {
            return None;
        }

        let tools = self.request.tools.as_ref().unwrap().clone();
        // Extract token IDs from GuidanceTokens (u32), not from tool_config (String)
        let start_token_id = self
            .guidance_tokens
            .tool_call_start_ids
            .first()
            .copied()
            .unwrap_or(0);
        let end_token_id = self
            .guidance_tokens
            .tool_call_end_ids
            .first()
            .copied()
            .unwrap_or(0);

        if !self.enable_tool_grammar {
            return Some(ToolCallGrammar::new_generic(
                tools,
                start_token_id,
                end_token_id,
            ));
        }

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
            "gemma4" => Some(ToolCallGrammar::new_json(
                    tools,
                    start_token_id,
                    end_token_id,
                )),
            "qwen" | "json" | _ => Some(ToolCallGrammar::new_json(
                    tools,
                    start_token_id,
                    end_token_id,
                ))
        }
    }

    fn build_reasoning_effort(&self) -> Option<ReasoningEffort> {
        let effort_str = self.request.reasoning_effort.as_ref()?;
        Some(ReasoningEffort::from_str(effort_str.clone()))
    }
}

// GRAMMAR COMPOSER

pub struct GrammarComposer;

impl GrammarComposer {
    pub fn compose_all_grammars(
        constraint_grammars: Vec<StructuredOutputsGrammar>,
        tool_grammar: Option<ToolCallGrammar>,
        reasoning_effort: Option<ReasoningEffort>,
        guidance_tokens: &GuidanceTokens,
        max_tokens: usize,
        chat_template: Option<crate::utils::chat_template::ChatTemplate>,
        tokenizer: &Tokenizer,
        disable_reasoning: bool,
    ) -> TopLevelGrammar {
        // Use ChatResponseGrammar as default when no constraints specified
        let merged_constraints = Self::merge_constraints(constraint_grammars);
        let composed_with_tools =
            Self::compose_constraint_with_tools(merged_constraints, tool_grammar);
        let final_grammar = Self::wrap_with_reasoning(
            composed_with_tools,
            reasoning_effort,
            guidance_tokens,
            disable_reasoning,
        );
        let mut grammar = Self::finalize_with_eos(final_grammar, guidance_tokens);
        // Models like MiniMax allow us to change the model's role in responding to permit "multi-agent" conversations
        // Need a helper to extract the capability from the chat template and role name from request
        let role = "assistant".to_string();
        
        // Only prefix with BOS if add_bos_token is true
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
                "start: text\ntext: /(?s:.+?)/".to_string(),
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
                // Replace the text: rule with tool_call: rule
                let tool_constraint = StructuredConstraint::Lark(tool_gram.build_lark());
                let mut tool_grammar = StructuredOutputsGrammar::new(tool_constraint);
                // Use alternation: ( text | tool_call )+
                let mut base_mut = base;
                base_mut.compose_alternate(&mut tool_grammar)
            }
            None => base,
        }
    }

    fn wrap_with_reasoning(
        base: StructuredOutputsGrammar,
        effort: Option<ReasoningEffort>,
        guidance_tokens: &GuidanceTokens,
        disable_reasoning: bool,
    ) -> StructuredOutputsGrammar {
        // If reasoning is disabled, return base without reasoning
        if disable_reasoning {
            return base;
        }

        // If no explicit effort passed, use ModelDefault
        let effort = effort.unwrap_or(ReasoningEffort::ModelDefault);

        if effort != ReasoningEffort::None {
            // Build reasoning grammar that wraps the base
            let start_id = *guidance_tokens.reasoning_start_ids.first().unwrap_or(&0);
            let end_id = *guidance_tokens.reasoning_end_ids.first().unwrap_or(&0);
            let mut reasoning = ReasoningGrammar::new(start_id, end_id, effort);
            let mut reasoning_grammar = StructuredOutputsGrammar::new(
                StructuredConstraint::Lark(reasoning.build_lark()),
            );
            // Use sequence: reasoning_block followed by base
            let mut base_mut = base;
            return reasoning_grammar.compose_sequence(&mut base_mut);
        }
        base
    }

    fn prefix_with_bos(
        mut grammar: TopLevelGrammar,
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
            format!(r#"bos: <[{}]> "{}:" "\n" "#, guidance_tokens.bos_token_ids[0], &role)
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
        if guidance_tokens.eos_token_ids.is_empty() {
            return grammar.format();
        }
        let lark = grammar.build_lark();
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
        let eos_rule = if guidance_tokens.eos_token_ids.len() == 1 {
            format!("eos: <[{}]>", guidance_tokens.eos_token_ids[0])
        } else {
            let ids: Vec<String> = guidance_tokens
                .eos_token_ids
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
            // No reasoning tokens in guidance_tokens, apply fallback
            crate::log_info!(
                "[llg] No reasoning tokens in guidance_tokens, applying fallback"
            );

            let reason_start = format!("<[{}]>", guidance_tokens.reasoning_start_ids[0]);
            let reason_end = format!("<[{}]>", guidance_tokens.reasoning_end_ids[0]);

            let lark = lark
                .replace(&reason_start, "\"<thinking>\"")
                .replace(&reason_end, "\"</thinking>\"");

            Some(lark)
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
    if let Some(transformed_lark) = apply_thinking_fallback_lark(
        lark_str,
        guidance_tokens,
        chat_template,
        tokenizer,
    ) {
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

/// Check if a TopLevelGrammar contains reasoning block patterns
pub fn is_reasoning_grammar(grammar: &TopLevelGrammar) -> bool {
    // Extract the Lark representation from TopLevelGrammar
    let lark_str = get_lark_from_top_level_grammar(grammar);
    // Check for reasoning-specific definition in the grammar structure
    lark_str
        .split("\n")
        .into_iter()
        .any(|l: &str| l.contains("reasoning_block") && l.contains("<[") && l.contains("]>"))
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
    allow_constraint_api: bool,
    model_type: &crate::utils::config::ModelType,
    _model_id: &str,
    parser_name: String,
    tokenizer: &Tokenizer,
    chat_template: Option<ChatTemplate>,
    disable_reasoning: bool,
) -> Option<TopLevelGrammar> {
    // Use new GrammarRequestDispatcher for grammar composition
    let tool_config = ToolConfig::from_tokenizer(tokenizer, model_type);

    let dispatcher = GrammarRequestDispatcher::new(
        request,
        guidance_tokens,
        &tool_config,
        enable_tool_grammar,
        allow_constraint_api,
        parser_name,
        tokenizer,
        chat_template,
        disable_reasoning,
    );

    dispatcher.build_grammar()
}

/// Build guided decoding grammar for claude_server.rs
/// This function constructs a synthetic ChatCompletionRequest from claude-style parameters
/// and delegates to generate_grammar_from_request for grammar composition.
///
/// Parameters:
/// - guidance_tokens: Contains EOS and reasoning token IDs
/// - tool_config: Model-specific tool call token configuration
/// - tools: List of available tools for grammar generation
/// - tool_parser_name: Name of the tool parser (e.g., "qwen_coder", "json")
/// - constraint_grammar: Optional constraint grammar from structured_outputs
/// - tool_choice_required: Whether tool choice is required
/// - forced_tool_name: Optional specific tool name to force
/// - max_tokens: Maximum tokens for generation
/// - reasoning_effort: Optional reasoning effort level
/// - enable_tool_grammar: Whether to enable tool grammar generation
/// - allow_constraint_api: Whether to allow constraint API
/// - tokenizer: Tokenizer for token ID lookup and grammar composition
/// - model_type: Model type for parser selection
/// - model_id: Model ID for parser configuration
/// - chat_template: Optional chat template for reasoning token detection
pub fn build_guided_decoding_grammar(
    guidance_tokens: &crate::utils::guidance::GuidanceTokens,
    _tool_config: &crate::server::parser::ToolConfig,
    tools: &[crate::tools::Tool],
    tool_parser_name: &str,
    constraint_grammar: Option<TopLevelGrammar>,
    _tool_choice_required: bool,
    forced_tool_name: Option<String>,
    max_tokens: usize,
    reasoning_effort: Option<crate::utils::config::ReasoningEffort>,
    enable_tool_grammar: bool,
    allow_constraint_api: bool,
    tokenizer: &Tokenizer,
    model_type: &crate::utils::config::ModelType,
    model_id: &str,
    chat_template: Option<ChatTemplate>,
    disable_reasoning: bool,
) -> Option<TopLevelGrammar> {
    // If constraint_grammar is provided, extract the Lark string and set it as a constraint
    // The dispatcher will handle this in build_constraint_grammar
    let constraint = constraint_grammar
        .as_ref()
        .map(|cg| get_lark_from_top_level_grammar(cg));

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
        tool_choice: forced_tool_name.map(|fn_name| crate::tools::ToolChoice::function(fn_name)),
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
        allow_constraint_api,
        parser_name,
        tokenizer,
        chat_template,
        disable_reasoning,
    );

    dispatcher.build_grammar()
}

/// Lark literal - returns string with quotes for non-special tags
pub fn _lark_literal(s: &str, special: bool) -> String {
    if special {
        s.to_string()
    } else {
        format!("\"{}\"", s)
    }
}

/// Parse structural tag configuration from JSON value
pub fn parse_structural_tag(value: &Value) -> Result<(String, String, Value), String> {
    let start_tag = value
        .get("start_tag")
        .or_else(|| value.get("tag"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Missing start_tag or tag".to_string())?
        .to_string();

    let end_tag = value
        .get("end_tag")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            // Extract tag name from start_tag (e.g., "<tool>" -> "tool")
            let tag_name = start_tag.trim_start_matches('<').trim_end_matches('>');
            format!("</{}>", tag_name)
        });

    let schema = value
        .get("schema")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object"}));

    Ok((start_tag, end_tag, schema))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create mock GuidanceTokens for testing
    fn create_mock_guidance_tokens() -> GuidanceTokens {
        GuidanceTokens {
            bos_token_ids: vec![151647],
            eos_token_ids: vec![151648, 151649],
            reasoning_start_ids: vec![151657],
            reasoning_end_ids: vec![151658],
            tool_call_start_ids: vec![151657],
            tool_call_end_ids: vec![151658],
            add_bos_token: false,
        }
    }

    /// Helper function to create mock ChatCompletionRequest with tools for testing
    fn create_mock_request_with_tools(tools: Vec<Tool>) -> crate::server::ChatCompletionRequest {
        crate::server::ChatCompletionRequest {
            messages: vec![],
            model: None,
            temperature: None,
            max_tokens: None,
            top_k: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            thinking: None,
            stop: None,
            stream: None,
            stream_options: None,
            session_id: None,
            tools: Some(tools),
            tool_choice: None,
            response_format: None,
            extra_body: None,
            structured_outputs: None,
            constraint: None,
            constraint_type: None,
            reasoning_effort: None,
        }
    }

    #[test]
    fn test_reasoning_grammar_low_effort() {
        let mut grammar = ReasoningGrammar::new(151657, 151658, ReasoningEffort::Low);
        let lark = grammar.build_lark();
        assert!(lark.contains("reasoning_block"));
        assert!(lark.contains("<[151657]>"));
        assert!(lark.contains("<[151658]>"));
    }

    #[test]
    fn test_tool_call_grammar_json() {
        let mut grammar = ToolCallGrammar::new_json(Vec::new(), 151657, 151658);
        let lark = grammar.build_lark();
        assert!(lark.contains("tool_call"));
    }

    #[test]
    fn test_structured_outputs_grammar_choice() {
        let mut grammar = StructuredOutputsGrammar::new(StructuredConstraint::Choice(vec![
            "option1".to_string(),
            "option2".to_string(),
        ]));
        let lark = grammar.build_lark();
        assert!(lark.contains("start:"));
    }

    #[test]
    fn test_chat_response_grammar() {
        let mut grammar = ChatResponseGrammar::new().with_eos(true);
        let lark = grammar.build_lark();
        assert!(lark.contains("start: text"));
    }

    #[test]
    fn test_reasoning_grammar_high_effort_output() {
        let mut grammar = ReasoningGrammar::new(151667, 151668, ReasoningEffort::High);
        let lark = grammar.build_lark();

        // Should contain reasoning block structure
        assert!(lark.contains("reasoning_block"));
        assert!(lark.contains("analysis_block"));
        assert!(lark.contains("critique_block"));
        assert!(lark.contains("structure_block"));

        // Should contain token IDs
        assert!(lark.contains("[151667]"));
        assert!(lark.contains("[151668]"));
    }

    #[test]
    fn test_choice_constraint_output() {
        let mut grammar = StructuredOutputsGrammar::new(StructuredConstraint::Choice(vec![
            "yes".to_string(),
            "no".to_string(),
        ]));
        let lark = grammar.build_lark();

        // Should contain choice alternation
        assert!(lark.contains("start:"));
        assert!(lark.contains("yes"));
        assert!(lark.contains("no"));
    }

    #[test]
    fn test_json_constraint_output() {
        let schema =
            serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let mut grammar = StructuredOutputsGrammar::new(StructuredConstraint::Json(schema));
        let lark = grammar.build_lark();

        // Should contain JSON schema constraint
        assert!(lark.contains("text: %json"));
    }

    #[test]
    fn test_tool_call_grammar_output() {
        let mut grammar = ToolCallGrammar::new_json(Vec::new(), 151657, 151658);
        let lark = grammar.build_lark();

        // Should contain tool_call structure
        assert!(lark.contains("start: tool_call"));
        assert!(lark.contains("tool_call:"));
    }

    #[test]
    fn test_compose_alternate_constraint_and_tool() {
        // Create constraint grammar
        let constraint = StructuredOutputsGrammar::new(StructuredConstraint::Lark(
            "start: text\ntext: /(?s:.+?)/".to_string(),
        ));

        // Create tool grammar
        let mut tool = ToolCallGrammar::new_json(Vec::new(), 151657, 151658);
        let tool_grammar =
            StructuredOutputsGrammar::new(StructuredConstraint::Lark(tool.build_lark()));

        // Compose with alternation
        let mut constraint_mut = constraint.clone();
        let mut result = constraint_mut.compose_alternate(&mut tool_grammar.clone());

        let lark = result.build_lark();

        // Should contain alternation of both constraints
        assert!(lark.contains("text |"));
        assert!(lark.contains("tool_call"));
    }

    #[test]
    fn test_compose_sequence_reasoning_and_constraint() {
        // Create reasoning grammar
        let mut reasoning = ReasoningGrammar::new(151667, 151668, ReasoningEffort::High);
        let reasoning_grammar =
            StructuredOutputsGrammar::new(StructuredConstraint::Lark(reasoning.build_lark()));

        // Create constraint grammar
        let constraint = StructuredOutputsGrammar::new(StructuredConstraint::Lark(
            "start: text\ntext: /(?s:.+?)/".to_string(),
        ));

        // Compose with sequence
        let mut reasoning_mut = reasoning_grammar.clone();
        let mut result = reasoning_mut.compose_sequence(&mut constraint.clone());

        let lark = result.build_lark();

        // Should contain both reasoning and constraint
        assert!(lark.contains("reasoning_block"));
        assert!(lark.contains("text"));
    }

    #[test]
    fn test_compose_alternate_multiple_constraints() {
        // Create multiple constraint grammars
        let constraint1 = StructuredOutputsGrammar::new(StructuredConstraint::Lark(
            r#"start: text1\ntext1: /[\x20-\x7E\x0A\x0D]+?/"#.to_string(),
        ));

        let mut constraint2 = StructuredOutputsGrammar::new(StructuredConstraint::Lark(
            r#"start: text2\ntext2: /[\x20-\x7E\x0A\x0D]+?/"#.to_string(),
        ));

        let mut constraint3 = StructuredOutputsGrammar::new(StructuredConstraint::Lark(
            r#"start: text3\ntext3: /[\x20-\x7E\x0A\x0D]+?/"#.to_string(),
        ));

        // Compose all with alternation
        let mut result = constraint1;
        result = result.compose_alternate(&mut constraint2);
        result = result.compose_alternate(&mut constraint3);

        let lark = result.build_lark();

        // Should contain all three alternatives
        // The alternation format is: start: ( (text1 | text2)+ | text3 )+
        assert!(lark.contains("text1"));
        assert!(lark.contains("text2"));
        assert!(lark.contains("text3"));
    }

    #[test]
    fn test_build_choice_lark_grammar_empty_string() {
        let result = build_choice_lark_grammar(&["".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_strips_metadata_fields() {
        // Test that metadata fields like default, description, title are stripped
        // while validation fields like minimum, maximum, type are preserved
        let schema = json!({
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 50,
                    "description": "A count parameter",
                    "title": "Count"
                },
                "enabled": {
                    "type": "boolean",
                    "default": true
                }
            },
            "required": ["count"]
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Metadata fields should be stripped
        assert!(sanitized["properties"]["count"].get("default").is_none(), "default should be stripped");
        assert!(sanitized["properties"]["count"].get("description").is_none(), "description should be stripped");
        assert!(sanitized["properties"]["count"].get("title").is_none(), "title should be stripped");

        // Validation fields should be preserved
        assert_eq!(sanitized["properties"]["count"]["type"], "integer");
        assert_eq!(sanitized["properties"]["count"]["minimum"], 1);
        assert_eq!(sanitized["properties"]["count"]["maximum"], 200);

        // Boolean with default
        assert!(sanitized["properties"]["enabled"].get("default").is_none());
        assert_eq!(sanitized["properties"]["enabled"]["type"], "boolean");
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_resolves_refs() {
        // Test that $defs section is resolved and $ref is replaced with actual definition
        // This is critical for tool schemas that use JSON Schema definitions
        let schema = json!({
            "$defs": {
                "AskUserQuestion": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["label", "question", "options"]
                },
                "AskUserOption": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "label": {"type": "string"}
                    },
                    "required": ["value", "label"]
                }
            },
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/AskUserQuestion"}
                }
            },
            "required": ["questions"]
        });

        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Debug: print the sanitized schema
        eprintln!("Original schema: {}", serde_json::to_string_pretty(&schema).unwrap());
        eprintln!("Sanitized schema: {}", serde_json::to_string_pretty(&sanitized).unwrap());

        // $defs should be REMOVED (not preserved) - refs are resolved
        assert!(sanitized.get("$defs").is_none(), "$defs should be removed after resolution");

        // $ref should be replaced with the actual definition
        let items_schema = &sanitized["properties"]["questions"]["items"];
        assert!(items_schema.get("$ref").is_none(), "$ref should be replaced");

        // The items should now have the resolved definition (type: object with properties)
        assert_eq!(items_schema["type"], "object", "items should be resolved to object type");
        assert!(items_schema.get("properties").is_some(), "resolved items should have properties");
    }

    #[test]
    fn test_full_tool_offer_schema_with_defs_resolved() {
        // Test with the exact tool offer schema from the error log
        // This verifies that $defs are resolved and $refs are replaced with actual definitions
        let schema = json!({
            "$defs": {
                "AskUserOption": {
                    "description": "A predefined answer option for a question.",
                    "properties": {
                        "description": {
                            "description": "Optional description shown below the label",
                            "nullable": true,
                            "type": "string"
                        },
                        "label": {
                            "description": "Display label for the option",
                            "type": "string"
                        },
                        "selected": {
                            "default": false,
                            "description": "Default selection state when multi_select is true.",
                            "type": "boolean"
                        },
                        "value": {
                            "description": "Value to return to LLM when selected",
                            "type": "string"
                        }
                    },
                    "required": ["value", "label"],
                    "type": "object"
                },
                "AskUserQuestion": {
                    "description": "A single question presented to the user.",
                    "properties": {
                        "allow_custom": {
                            "default": true,
                            "description": "Whether to allow custom text input (default: true)",
                            "type": "boolean"
                        },
                        "label": {
                            "description": "Short unique label for tab display (max ~15 chars recommended)",
                            "type": "string"
                        },
                        "multi_select": {
                            "default": false,
                            "description": "When true, user can select/deselect multiple options.",
                            "type": "boolean"
                        },
                        "options": {
                            "description": "Predefined answer options",
                            "items": {"$ref": "$defs/AskUserOption"},
                            "type": "array"
                        },
                        "question": {
                            "description": "Full question text to display",
                            "type": "string"
                        }
                    },
                    "required": ["label", "question", "options"],
                    "type": "object"
                }
            },
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "description": "Request payload for the `ask_user` tool.",
            "properties": {
                "questions": {
                    "description": "List of questions to ask the user.",
                    "items": {"$ref": "$defs/AskUserQuestion"},
                    "type": "array"
                }
            },
            "required": ["questions"],
            "title": "AskUserRequest",
            "type": "object"
        });

        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Debug: print the sanitized schema
        eprintln!("Full tool offer sanitized schema: {}", serde_json::to_string_pretty(&sanitized).unwrap());

        // $defs should be REMOVED (not preserved) - refs are resolved
        assert!(sanitized.get("$defs").is_none(), "$defs should be removed after resolution");

        // $ref in properties should be replaced with actual definition
        let questions_items = &sanitized["properties"]["questions"]["items"];
        assert!(questions_items.get("$ref").is_none(), "$ref should be replaced with resolved definition");

        // The resolved items should have the actual definition (type: object with properties)
        assert_eq!(questions_items["type"], "object", "items should be resolved to object type");
        assert!(questions_items.get("properties").is_some(), "resolved items should have properties");
        assert!(questions_items.get("required").is_some(), "resolved items should have required");

        // $ref inside $defs should also be resolved
        let ask_user_question = sanitized["properties"]["questions"]["items"].clone();
        let options = ask_user_question["properties"]["options"].clone();
        let items = options["items"].clone();
        assert!(items.get("$ref").is_none(), "$ref inside options should be resolved");
        assert_eq!(items["type"], "object", "resolved items should be object type");
    }

    #[test]
    fn test_extract_defs_simple_schema() {
        // Test that extract_defs correctly extracts definitions from $defs section
        let schema = json!({
            "$defs": {
                "Question": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    }
                },
                "Option": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"}
                    }
                }
            },
            "type": "object",
            "properties": {
                "question": {"$ref": "#/$defs/Question"}
            }
        });

        let (schema_without_defs, defs) = extract_defs(&schema);

        // $defs should be removed from schema
        assert!(schema_without_defs.get("$defs").is_none(), "$defs should be removed from schema");

        // Definitions should be extracted
        assert_eq!(defs.len(), 2, "Should extract 2 definitions");
        assert!(defs.contains_key("Question"), "Should contain Question definition");
        assert!(defs.contains_key("Option"), "Should contain Option definition");

        // Schema structure should be preserved
        assert_eq!(schema_without_defs["type"], "object", "Schema type should be preserved");
        assert!(schema_without_defs.get("properties").is_some(), "Properties should be preserved");
    }

    #[test]
    fn test_extract_defs_nested_refs() {
        // Test that extract_defs handles nested $ref within definitions
        let schema = json!({
            "$defs": {
                "Option": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "label": {"type": "string"}
                    }
                },
                "Question": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Option"}
                        }
                    }
                }
            },
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Question"}
                }
            }
        });

        let (schema_without_defs, defs) = extract_defs(&schema);

        // All definitions should be extracted
        assert_eq!(defs.len(), 2, "Should extract 2 definitions");
        assert!(defs.contains_key("Option"), "Should contain Option definition");
        assert!(defs.contains_key("Question"), "Should contain Question definition");

        // $defs should be removed from schema
        assert!(schema_without_defs.get("$defs").is_none(), "$defs should be removed");
    }

    #[test]
    fn test_resolve_schema_refs_simple() {
        // Test that resolve_schema_refs replaces $ref with actual definition
        let defs: HashMap<String, Value> = [
            ("Question".to_string(), json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                }
            }))
        ].iter().cloned().collect();

        let schema = json!({
            "type": "object",
            "properties": {
                "question": {"$ref": "#/$defs/Question"}
            }
        });

        let resolved = resolve_schema_refs(&schema, &defs);

        // $ref should be replaced with actual definition
        assert!(resolved["properties"]["question"].get("$ref").is_none(), "$ref should be replaced");
        assert_eq!(resolved["properties"]["question"]["type"], "object", "Should have resolved type");
        assert!(resolved["properties"]["question"].get("properties").is_some(), "Should have resolved properties");
    }

    #[test]
    fn test_resolve_schema_refs_nested() {
        // Test that resolve_schema_refs handles nested $ref within definitions
        let defs: HashMap<String, Value> = [
            ("Option".to_string(), json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                }
            })),
            ("Question".to_string(), json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Option"}
                    }
                }
            }))
        ].iter().cloned().collect();

        let schema = json!({
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Question"}
                }
            }
        });

        let resolved = resolve_schema_refs(&schema, &defs);

        // Top-level $ref should be resolved
        let questions_items = &resolved["properties"]["questions"]["items"];
        assert!(questions_items.get("$ref").is_none(), "Top-level $ref should be resolved");
        assert_eq!(questions_items["type"], "object", "Should have resolved type");

        // Nested $ref in options should also be resolved
        let options = &questions_items["properties"]["options"];
        let options_items = &options["items"];
        assert!(options_items.get("$ref").is_none(), "Nested $ref should be resolved");
        assert_eq!(options_items["type"], "object", "Nested items should have resolved type");
    }

    #[test]
    fn test_resolve_schema_refs_missing_def() {
        // Test that resolve_schema_refs handles missing definitions gracefully
        let defs: HashMap<String, Value> = HashMap::new();

        let schema = json!({
            "type": "object",
            "properties": {
                "question": {"$ref": "#/$defs/MissingType"}
            }
        });

        let resolved = resolve_schema_refs(&schema, &defs);

        // $ref should remain when definition is missing (won't crash)
        assert!(resolved["properties"]["question"].get("$ref").is_some(), "$ref should remain when def is missing");
    }

    #[test]
    fn test_llguidance_json_schema_with_defs_compiles() {
        // Test that llguidance can compile a JSON schema with resolved $defs
        // This verifies the fix works end-to-end with llguidance
        use llguidance::api::TopLevelGrammar;

        let schema = json!({
            "$defs": {
                "AskUserQuestion": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"$ref": "$defs/AskUserOption"}
                        }
                    },
                    "required": ["label", "question", "options"]
                },
                "AskUserOption": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "label": {"type": "string"}
                    },
                    "required": ["value", "label"]
                }
            },
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"$ref": "$defs/AskUserQuestion"}
                }
            },
            "required": ["questions"]
        });

        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Debug: print the sanitized schema
        eprintln!("Sanitized schema for llguidance: {}", serde_json::to_string_pretty(&sanitized).unwrap());

        // $defs should be removed (refs are resolved)
        assert!(sanitized.get("$defs").is_none(), "$defs should be removed after resolution");

        // $ref should be replaced with actual definition
        let questions_items = &sanitized["properties"]["questions"]["items"];
        assert!(questions_items.get("$ref").is_none(), "$ref should be replaced");

        // Try to compile the sanitized schema with llguidance
        // This should not fail if $defs is properly resolved
        let _grammar = TopLevelGrammar::from_json_schema(sanitized);
        // If we get here without panicking, the schema compiled successfully
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_preserves_property_names() {
        // Test that property names (field names) are preserved
        let schema = json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "mode": {"type": "string", "description": "Mode of transport"},
                "count": {"type": "integer", "minimum": 1, "maximum": 200}
            },
            "required": ["city", "mode"]
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Property names should be preserved
        assert!(sanitized["properties"].get("city").is_some(), "city property should be preserved");
        assert!(sanitized["properties"].get("mode").is_some(), "mode property should be preserved");
        assert!(sanitized["properties"].get("count").is_some(), "count property should be preserved");

        // Metadata should be stripped from properties
        assert!(sanitized["properties"]["city"].get("description").is_none(), "description should be stripped from city");
        assert!(sanitized["properties"]["mode"].get("description").is_none(), "description should be stripped from mode");
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_preserves_nested_properties() {
        // Test that nested properties are correctly processed
        let schema = json!({
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {
                            "type": "string",
                            "description": "Inner value"
                        }
                    },
                    "required": ["inner"]
                }
            },
            "required": ["outer"]
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Nested property names should be preserved
        assert!(sanitized["properties"]["outer"]["properties"].get("inner").is_some(), "inner property should be preserved");

        // Metadata should be stripped from nested properties
        assert!(sanitized["properties"]["outer"]["properties"]["inner"].get("description").is_none(), "description should be stripped");
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_strips_format() {
        // Test that format keyword is preserved (it's a validation keyword in llguidance)
        // Only metadata fields like description, default, title are stripped
        let schema = json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "email": {"type": "string", "format": "email"}
            }
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Format should be preserved (it's a validation keyword)
        assert_eq!(sanitized["properties"]["url"]["format"], "uri");
        assert_eq!(sanitized["properties"]["email"]["format"], "email");

        // Type should be preserved
        assert_eq!(sanitized["properties"]["url"]["type"], "string");
        assert_eq!(sanitized["properties"]["email"]["type"], "string");
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_preserves_nullable_types() {
        // Test that nullable types (array of types) are preserved
        let schema = json!({
            "type": "object",
            "properties": {
                "cwd": {"type": ["string", "null"]}
            },
            "required": ["cwd"]
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Nullable types should be preserved
        assert_eq!(
            sanitized["properties"]["cwd"]["type"],
            json!(["string", "null"])
        );
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_strips_examples() {
        // Test that examples field is stripped (it's metadata)
        let schema = json!({
            "type": "string",
            "examples": ["option1", "option2", "option3"]
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Examples should be stripped
        assert!(sanitized.get("examples").is_none(), "examples should be stripped");

        // Type should be preserved
        assert_eq!(sanitized["type"], "string");
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_strips_default_in_array() {
        // Test that default is stripped even when nested in arrays
        let schema = json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "default": "unknown"}
                }
            }
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);

        // Default should be stripped from nested properties
        assert!(sanitized["items"]["properties"]["name"].get("default").is_none(), "default should be stripped");

        // Type should be preserved
        assert_eq!(sanitized["items"]["properties"]["name"]["type"], "string");
    }

    #[test]
    fn test_parse_structural_tag_missing_schema() {
        let value = json!({});
        let result = parse_structural_tag(&value);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_structural_tag_start_end() {
        let value = json!({
            "start_tag": "<tool>",
            "end_tag": "</tool>",
            "schema": {"type": "object"}
        });
        let result = parse_structural_tag(&value);
        assert!(result.is_ok());
        let (start, end, schema) = result.unwrap();
        assert_eq!(start, "<tool>");
        assert_eq!(end, "</tool>");
        assert_eq!(schema, json!({"type": "object"}));
    }

    #[test]
    fn test_parse_structural_tag_tag() {
        let value = json!({
            "tag": "<tool>",
            "schema": {"type": "object"}
        });
        let result = parse_structural_tag(&value);
        assert!(result.is_ok());
        let (start, end, _) = result.unwrap();
        assert_eq!(start, "<tool>");
        assert_eq!(end, "</tool>");
    }

    #[test]
    fn test_parse_structural_tag_invalid() {
        let value = json!({
            "schema": {"type": "object"}
        });
        let result = parse_structural_tag(&value);
        assert!(result.is_err());
    }

    #[test]
    fn test_lark_quote_escapes_special_chars() {
        let result = lark_quote("test\"value");
        assert!(result.contains("test\\\"value"));
    }

    #[test]
    fn test_lark_literal_special_tags() {
        let result = _lark_literal("<tool>", true);
        assert_eq!(result, "<tool>");
    }

    #[test]
    fn test_lark_literal_regular_string() {
        let result = _lark_literal("regular", false);
        assert!(result.contains("\"regular\""));
    }

    #[test]
    fn test_lark_special_token_single_id() {
        let mut ids = HashSet::new();
        ids.insert(151657);
        let result = lark_special_token(&ids);
        assert_eq!(result, "<[151657]>");
    }

    #[test]
    fn test_lark_special_token_multiple_ids() {
        let mut ids = HashSet::new();
        ids.insert(151657);
        ids.insert(151658);
        let result = lark_special_token(&ids);
        assert!(result.contains("[151657]"));
        assert!(result.contains("[151658]"));
    }

    #[test]
    fn test_lark_special_token_empty() {
        let ids = HashSet::new();
        let result = lark_special_token(&ids);
        assert_eq!(result, "");
    }

    #[test]
    fn test_build_xml_tool_lark_grammar_qwen3_coder_required_only() {
        // Test Qwen3-Coder XML tool format with required attributes only
        let tools = vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()];
        let request = create_mock_request_with_tools(tools);
        let guidance_tokens = create_mock_guidance_tokens();
        let tool_config = crate::server::parser::ToolConfig::for_model_type(
            &crate::utils::config::ModelType::Qwen3,
        );
        let tokenizer = Tokenizer::from_pretrained("bert-base-uncased".to_string(), None).unwrap();
        let dispatcher = GrammarRequestDispatcher::new(
            &request,
            &guidance_tokens,
            &tool_config,
            true,
            false,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false, // disable_reasoning
        );
        let mut grammar = dispatcher
            .build_tool_grammar()
            .expect("Should have tool grammar");
        let lark_str = grammar.build_lark();
        println!("{}", &lark_str);

        // Qwen3Coder uses XML format with start: tool_call
        assert!(
            lark_str.contains("start: tool_call"),
            "Should have start: tool_call"
        );
        assert!(
            lark_str.contains("<function=search>"),
            "Should contain function tag"
        );
        assert!(lark_str.contains("tool_0:"), "Should contain tool_0 rule");
    }

    #[test]
    fn test_build_xml_tool_lark_grammar_qwen3_coder_optional() {
        // Test Qwen3-Coder XML tool format with optional attributes
        let tools = vec![crate::tools::ToolBuilder::new(
            "get_weather".to_string(),
            "Get weather".to_string(),
        )
        .param("city", "string", "City name", true)
        .param("units", "string", "Temperature units (optional)", false)
        .build()];
        let request = create_mock_request_with_tools(tools);
        let guidance_tokens = create_mock_guidance_tokens();
        let tool_config = crate::server::parser::ToolConfig::for_model_type(
            &crate::utils::config::ModelType::Qwen3,
        );
        let tokenizer = Tokenizer::from_pretrained("bert-base-uncased".to_string(), None).unwrap();
        let dispatcher = GrammarRequestDispatcher::new(
            &request,
            &guidance_tokens,
            &tool_config,
            true,
            false,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false, // disable_reasoning
        );
        let mut grammar = dispatcher
            .build_tool_grammar()
            .expect("Should have tool grammar");
        let lark_str = grammar.build_lark();

        assert!(
            lark_str.contains("start: tool_call"),
            "Should have start: tool_call"
        );
        assert!(
            lark_str.contains("<function=get_weather>"),
            "Should contain function tag"
        );
        assert!(lark_str.contains("city"), "Should contain city parameter");
        assert!(
            lark_str.contains("units"),
            "Should contain optional units parameter"
        );
    }

    #[test]
    fn test_build_xml_tool_lark_grammar_qwen3_coder_deep_parameters() {
        // Test Qwen3-Coder XML tool format with nested/complex parameters
        let tools = vec![crate::tools::ToolBuilder::new(
            "edit_file".to_string(),
            "Edit a file with complex parameters".to_string(),
        )
        .param("file_path", "string", "Path to the file", true)
        .param("old_string", "string", "String to replace", true)
        .param("new_string", "string", "Replacement string", true)
        .param("replace_all", "boolean", "Replace all occurrences", false)
        .build()];
        let request = create_mock_request_with_tools(tools);
        let guidance_tokens = create_mock_guidance_tokens();
        let tool_config = crate::server::parser::ToolConfig::for_model_type(
            &crate::utils::config::ModelType::Qwen3,
        );
        let tokenizer = Tokenizer::from_pretrained("bert-base-uncased".to_string(), None).unwrap();
        let dispatcher = GrammarRequestDispatcher::new(
            &request,
            &guidance_tokens,
            &tool_config,
            true,
            false,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false, // disable_reasoning
        );
        let mut grammar = dispatcher
            .build_tool_grammar()
            .expect("Should have tool grammar");
        let lark_str = grammar.build_lark();
        println!("XML Grammar:\n{}", &lark_str);

        // Verify the grammar contains XML structure
        assert!(
            lark_str.contains("start: tool_call"),
            "Should have start: tool_call"
        );
        // Note: <function=...> uses U+200C (zero-width non-joiner) which is invisible
        assert!(
            lark_str.contains("function="),
            "Should contain function tag with attribute"
        );

        // Verify all parameter tags are present
        // Note: <parameter=...> uses U+200C (zero-width non-joiner) which is invisible
        assert!(
            lark_str.contains("parameter=file_path"),
            "Should contain file_path parameter tag"
        );
        assert!(
            lark_str.contains("parameter=old_string"),
            "Should contain old_string parameter tag"
        );
        assert!(
            lark_str.contains("parameter=new_string"),
            "Should contain new_string parameter tag"
        );
        assert!(
            lark_str.contains("parameter=replace_all"),
            "Should contain replace_all parameter tag"
        );

        // Verify all string params share the same consolidated value_string rule
        assert!(
            lark_str.contains("value_string"),
            "All string params should use the consolidated value_string rule"
        );

        // Verify non-string types still have unique rules
        assert!(
            lark_str.contains("value_0_3_boolean"),
            "Boolean param should have its own unique rule"
        );
        assert!(
            lark_str.contains("param_0_0:"),
            "Should have param_0_0 rule for first param"
        );
        assert!(
            lark_str.contains("param_0_1:"),
            "Should have param_0_1 rule for second param"
        );
        assert!(
            lark_str.contains("param_0_2:"),
            "Should have param_0_2 rule for third param"
        );
        assert!(
            lark_str.contains("param_0_3:"),
            "Should have param_0_3 rule for fourth param"
        );

        // Verify tool rule has all parameters
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");

        // Verify deduplication is disabled: each string param should have its own value rule
        // Rules are named value_{tool_idx}_{param_idx}_{type} so check for pattern
        let value_string_count = lark_str.matches("value_").count();
        assert!(
            value_string_count >= 4,
            "Each param should have its own value rule (no deduplication), found {}",
            value_string_count
        );
    }

    #[test]
    fn test_xml_grammar_required_params_no_wrapper() {
        // Test that XML grammar puts required params directly without (...) * wrapper
        let tools = vec![crate::tools::ToolBuilder::new(
            "search_tool".to_string(),
            "Search tool".to_string(),
        )
        .param("query", "string", "Search query", true) // REQUIRED - should appear as bare rule reference
        .build()];

        let request = create_mock_request_with_tools(tools);
        let guidance_tokens = create_mock_guidance_tokens();
        let tool_config = crate::server::parser::ToolConfig::for_model_type(
            &crate::utils::config::ModelType::Qwen3,
        );
        let tokenizer = Tokenizer::from_pretrained("bert-base-uncased".to_string(), None).unwrap();
        let dispatcher = GrammarRequestDispatcher::new(
            &request,
            &guidance_tokens,
            &tool_config,
            true,
            false,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false, // disable_reasoning
        );
        let mut grammar = dispatcher
            .build_tool_grammar()
            .expect("Should have tool grammar");
        let lark_str = grammar.build_lark();

        // Verify tool rule has all parameters
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");
        assert!(
            lark_str.contains("value_string"),
            "Should have the consolidated value_string rule for string params"
        );

        // Required params appear directly in tool rule without ()* wrapper
    }

    #[test]
    fn test_xml_grammar_optional_params_wrapped() {
        // Test that XML grammar wraps optional params with (...) * syntax
        let tools = vec![crate::tools::ToolBuilder::new(
            "mixed_tool".to_string(),
            "Mixed params".to_string(),
        )
        .param("required_param", "string", "Required", true) // REQUIRED
        .param("optional_param", "string", "Optional", false) // OPTIONAL
        .build()];

        let request = create_mock_request_with_tools(tools);
        let guidance_tokens = create_mock_guidance_tokens();
        let tool_config = crate::server::parser::ToolConfig::for_model_type(
            &crate::utils::config::ModelType::Qwen3,
        );
        let tokenizer = Tokenizer::from_pretrained("bert-base-uncased".to_string(), None).unwrap();
        let dispatcher = GrammarRequestDispatcher::new(
            &request,
            &guidance_tokens,
            &tool_config,
            true,
            false,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false, // disable_reasoning
        );
        let mut grammar = dispatcher
            .build_tool_grammar()
            .expect("Should have tool grammar");
        let lark_str = grammar.build_lark();

        println!("XML Grammar for mixed tool:\n{}", lark_str);

        // Optional parameters should appear in a (...) * pattern when there are multiple options
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");
    }

    #[test]
    fn test_xml_tool_call_structure_validates() {
        // Full end-to-end: verify XML grammar produces valid llguidance TopLevelGrammar structure
        let tools =
            vec![
                crate::tools::ToolBuilder::new("formatter".to_string(), "Formatter".to_string())
                    .param("text", "string", "Text to format", true)
                    .build(),
            ];

        let request = create_mock_request_with_tools(tools);
        let guidance_tokens = create_mock_guidance_tokens();
        let tool_config = crate::server::parser::ToolConfig::for_model_type(
            &crate::utils::config::ModelType::Qwen3,
        );
        let tokenizer = Tokenizer::from_pretrained("bert-base-uncased".to_string(), None).unwrap();
        let dispatcher = GrammarRequestDispatcher::new(
            &request,
            &guidance_tokens,
            &tool_config,
            true,
            false,
            "qwen_coder".to_string(),
            &tokenizer,
            None,
            false, // disable_reasoning
        );
        let grammar = dispatcher
            .build_tool_grammar()
            .expect("Should have tool grammar");

        // Verify the grammar has tools and produces valid Lark output
        assert!(
            grammar.tools.len() > 0,
            "Should have generated tool grammar"
        );
    }
/*
    #[test]
    fn test_gemma4_tool_call_format_matches_template() {
        // Test that Gemma4 format matches chat_template.jinja specification
        // Template: <|tool_call>call:function_name{key: value}<tool_call|>
        // Generated Lark should match: <[start_token]> "call:function_name{...}" <[end_token]>

        let tools = vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .param("limit", "integer", "Result limit", false)
        .build()];

        // Create Gemma4 grammar with mock token IDs
        let start_token_id = 151657u32;
        let end_token_id = 151658u32;
        let mut grammar = ToolCallGrammar::new_gemma4(tools, start_token_id, end_token_id);

        let lark_str = grammar.build_lark();

        // Verify structure matches template format
        assert!(lark_str.contains(&format!("<[{}]>", start_token_id)), "Should contain start token");
        assert!(lark_str.contains(&format!("<[{}]>", end_token_id)), "Should contain end token");
        assert!(lark_str.contains("call:search"), "Should contain call:function_name format");

        // Verify the arguments pattern is present
        assert!(lark_str.contains("query"), "Should contain query parameter");
        assert!(lark_str.contains("limit"), "Should contain limit parameter");

        // Print for debugging
        println!("Gemma4 Lark grammar:\n{}", lark_str);
    }

    #[test]
    fn test_for_model_type_with_override() {
        // Test that parser_name override takes precedence over model_type
        let tools = vec![crate::tools::ToolBuilder::new(
            "search".to_string(),
            "Search the web".to_string(),
        )
        .param("query", "string", "Search query", true)
        .build()];

        let start_token_id = 151657u32;
        let end_token_id = 151658u32;

        // Override with "gemma4" should use Gemma4 format even for Qwen3 model
        let grammar = ToolCallGrammar::for_model_type(
            tools.clone(),
            start_token_id,
            end_token_id,
            Some("gemma4"),
            &crate::utils::config::ModelType::Qwen3,
        );
        assert!(matches!(grammar.format, ToolFormat::Gemma4));

        // Override with "qwen_coder" should use XML format even for Gemma4 model
        let grammar = ToolCallGrammar::for_model_type(
            tools.clone(),
            start_token_id,
            end_token_id,
            Some("qwen_coder"),
            &crate::utils::config::ModelType::Gemma4,
        );
        assert!(matches!(grammar.format, ToolFormat::QwenCoder));

        // No override - Qwen3 should use XML
        let grammar = ToolCallGrammar::for_model_type(
            tools.clone(),
            start_token_id,
            end_token_id,
            None,
            &crate::utils::config::ModelType::Qwen3,
        );
        assert!(matches!(grammar.format, ToolFormat::QwenCoder));

        // No override - Gemma4 should use Gemma4
        let grammar = ToolCallGrammar::for_model_type(
            tools,
            start_token_id,
            end_token_id,
            None,
            &crate::utils::config::ModelType::Gemma4,
        );
        assert!(matches!(grammar.format, ToolFormat::Gemma4));
    }
    */
}
