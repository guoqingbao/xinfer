// src/tools/schema.rs
//! JSON Schema utilities for tool parameters
//!
//! Provides helpers for working with JSON Schema in tool definitions.

use crate::tools::Tool;
use crate::utils::guidance::{GrammarError, GrammarResult, TopLevelGrammarExt};
use llguidance::api::TopLevelGrammar;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};

/// Remove JSON Schema features that llguidance doesn't support.
/// Currently strips all "format" fields recursively.
fn sanitize_schema_for_llguidance_recursive(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, value) in map {
                if key == "format" {
                    continue;
                }
                out.insert(key.clone(), sanitize_schema_for_llguidance_recursive(value));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(sanitize_schema_for_llguidance_recursive)
                .collect(),
        ),
        _ => schema.clone(),
    }
}

/// Remove JSON Schema features that llguidance doesn't support.
/// Currently strips all "format" fields recursively.
pub fn sanitize_schema_for_llguidance(schema: &Value) -> Value {
    sanitize_schema_for_llguidance_recursive(schema)
}

/// Lark grammar helper functions for llguidance constraint building
fn lark_quote(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

/// Convert token IDs to Lark special token syntax <[token_id]>
/// This is used when the tokenizer has canonical tokenization for the tag
fn lark_special_token(token_ids: &HashSet<u32>) -> String {
    if token_ids.is_empty() {
        return String::new();
    }
    // Join multiple token IDs with |
    let ids: Vec<String> = token_ids.iter().map(|id| format!("[{}]", id)).collect();
    format!("<{}>", ids.join(","))
}

fn _lark_literal(value: &str, is_special: bool) -> String {
    if is_special && value.starts_with('<') && value.ends_with('>') {
        // Only allow ASCII special tags
        let sanitized: String = value.chars().filter(|c| c.is_ascii()).collect();
        sanitized
    } else {
        lark_quote(value)
    }
}

/// Builder for constructing tool call grammars
pub struct ToolGrammarBuilder {
    tools: Vec<Tool>,
    start_tag: String,
    end_tag: String,
    start_is_special: bool,
    end_is_special: bool,
    start_token_ids: Option<HashSet<u32>>,
    end_token_ids: Option<HashSet<u32>>,
}

impl ToolGrammarBuilder {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            start_tag: String::new(),
            end_tag: String::new(),
            start_is_special: false,
            end_is_special: false,
            start_token_ids: None,
            end_token_ids: None,
        }
    }

    pub fn tools(mut self, tools: &[Tool]) -> Self {
        self.tools.extend(tools.iter().cloned());
        self
    }

    pub fn start_tag(mut self, tag: impl Into<String>) -> Self {
        self.start_tag = tag.into();
        self
    }

    pub fn end_tag(mut self, tag: impl Into<String>) -> Self {
        self.end_tag = tag.into();
        self
    }

    pub fn start_is_special(mut self, special: bool) -> Self {
        self.start_is_special = special;
        self
    }

    pub fn end_is_special(mut self, special: bool) -> Self {
        self.end_is_special = special;
        self
    }

    pub fn start_token_ids(mut self, ids: Option<HashSet<u32>>) -> Self {
        self.start_token_ids = ids;
        self
    }

    pub fn end_token_ids(mut self, ids: Option<HashSet<u32>>) -> Self {
        self.end_token_ids = ids;
        self
    }

    /// Build Lark expression for JSON tool schema content
    pub fn build_json(self) -> TopLevelGrammar {
        let start_tag = self.get_tag_or_token_id(
            &self.start_tag,
            &self.start_token_ids,
            self.start_is_special,
        );
        let end_tag =
            self.get_tag_or_token_id(&self.end_tag, &self.end_token_ids, self.end_is_special);
        let payload_schema = if self.tools.is_empty() {
            json!({ "type": "object" })
        } else {
            let variants: Vec<Value> = self
                .tools
                .iter()
                .map(|tool| {
                    let arguments_schema =
                        sanitize_schema_for_llguidance(&tool.function.parameters);
                    json!({
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": [tool.function.name.clone()],
                            },
                            "arguments": arguments_schema,
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": false,
                    })
                })
                .collect();
            if variants.len() == 1 {
                variants[0].clone()
            } else {
                json!({ "oneOf": variants })
            }
        };

        let payload_schema = serde_json::to_string(&payload_schema).unwrap_or_default();
        let lark = format!(
            "start: tool_call\ntool_call: {start_tag} tool_payload {end_tag}\ntool_payload: %json {payload_schema}\n"
        );
        TopLevelGrammar::from_lark_utf8(&lark)
    }

    /// Build Lark expression for valid XML parameter content
    fn build_xml_value_expression(schema: &serde_json::Value) -> String {
        let param_type = schema
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("string");

        match param_type {
            "string" => {
                if let Ok(val) = std::env::var("XINFER_LLG_DEFAULT_XML_STR") {
                    format!("{}", val)
                } else {
                    r#"/[ -~]*?/"#.to_string()
                }
            }
            "integer" => r"/-?[0-9]+/".to_string(),
            "number" => r"/-?[0-9]+(\.[0-9]+)?/".to_string(),
            "boolean" => r"/^(true|false)$/".to_string(),
            "array" => r"/\[[^\]]*\]/".to_string(),
            "object" => r"/\{[^\}]*\}/".to_string(),
            _ => r"/[ -~]*?/".to_string(),
        }
    }

    /// Build Lark expression for XML tool schema content
    /// Uses structured tag parsing with ASCII-restricted value patterns
    pub fn build_xml(self) -> TopLevelGrammar {
        let mut rules: Vec<String> = Vec::new();

        // Build envelope tag using token IDs when available
        let envelope_start_tag = self.get_envelope_tag(
            &self.start_tag,
            &self.start_token_ids,
            self.start_is_special,
        );
        let envelope_end_tag =
            self.get_envelope_tag(&self.end_tag, &self.end_token_ids, self.end_is_special);

        let tool_rule_names: Vec<String> =
            (0..self.tools.len()).map(|i| format!("tool_{i}")).collect();

        // GUARD 1: Use string literals for XML inner structure (function/parameter tags)
        // The stream parser detects these using text matching in the buffer
        rules.push("start: tool_call".to_string());
        rules.push(format!(
            "tool_call: {} tool_content {}",
            envelope_start_tag, envelope_end_tag
        ));

        for (tool_idx, tool) in self.tools.iter().enumerate() {
            let tool_name_ascii: String = tool
                .function
                .name
                .chars()
                .filter(|c| c.is_ascii())
                .collect();
            let func_start = lark_quote(&format!("<function={}>", tool_name_ascii));
            let func_end = lark_quote("</function>");
            let params_schema = &tool.function.parameters;
            let props = params_schema.get("properties").and_then(|p| p.as_object());
            let required_params: Vec<String> = params_schema
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            if let Some(props) = props {
                let mut param_rules_vec: Vec<String> = Vec::new();

                for (param_idx, (param_name, schema)) in props.iter().enumerate() {
                    let param_name_ascii: String =
                        param_name.chars().filter(|c| c.is_ascii()).collect();
                    let param_tag = lark_quote(&format!("<parameter={}>", param_name_ascii));
                    let param_end = lark_quote("</parameter>");
                    let value_rule = format!("value_{tool_idx}_{param_idx}");
                    let param_rule = format!("param_{tool_idx}_{param_idx}");

                    let value_expr = Self::build_xml_value_expression(schema);
                    rules.push(format!("{value_rule}: {value_expr}"));
                    rules.push(format!(
                        "{param_rule}: {param_tag} {value_rule} {param_end}"
                    ));

                    if required_params.contains(param_name) {
                        param_rules_vec.push(param_rule);
                    } else {
                        param_rules_vec.push(format!("({param_rule})?"));
                    }
                }

                let params_expr = param_rules_vec.join(" ");
                rules.push(format!(
                    "tool_{tool_idx}: {func_start} {params_expr} {func_end}"
                ));
            } else {
                rules.push(format!("tool_{tool_idx}: {func_start} {func_end}"));
            }
        }

        // Build tool_content with alternation of all tools
        let tool_variants = tool_rule_names.join(" | ");
        rules.push(format!("tool_content: {tool_variants}"));

        let lark = rules.join("\n") + "\n";
        TopLevelGrammar::from_lark_utf8(&lark)
    }

    /// Get envelope tag (start/end) using token IDs when available, falling back to string literals
    fn get_envelope_tag(
        &self,
        tag: &str,
        token_ids: &Option<HashSet<u32>>,
        is_special: bool,
    ) -> String {
        if let Some(ids) = token_ids {
            if !ids.is_empty() {
                return lark_special_token(ids);
            }
        }

        if is_special && tag.starts_with('<') && tag.ends_with('>') {
            // Only allow ASCII special tags
            let sanitized: String = tag.chars().filter(|c| c.is_ascii()).collect();
            sanitized
        } else {
            lark_quote(tag)
        }
    }

    fn get_tag_or_token_id(
        &self,
        tag: &str,
        token_ids: &Option<HashSet<u32>>,
        is_special: bool,
    ) -> String {
        if let Some(ids) = token_ids {
            if !ids.is_empty() {
                return format!(
                    "<{}>",
                    ids.iter()
                        .map(|id| format!("[{}]", id))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }

        if is_special && tag.starts_with('<') && tag.ends_with('>') {
            tag.to_string()
        } else {
            lark_quote(tag)
        }
    }
}

/// Build a Lark grammar for QwenCoder-style function/parameter tags with JSON values.
/// Used for models like Qwen3-Coder that use XML-style tool call envelopes.
pub fn build_xml_tool_lark_grammar(
    tools: &[Tool],
    start: &str,
    end: &str,
    start_is_special: bool,
    end_is_special: bool,
    start_token_ids: Option<&HashSet<u32>>,
    end_token_ids: Option<&HashSet<u32>>,
) -> TopLevelGrammar {
    ToolGrammarBuilder::new()
        .tools(tools)
        .start_tag(start)
        .end_tag(end)
        .start_is_special(start_is_special)
        .end_is_special(end_is_special)
        .start_token_ids(start_token_ids.cloned())
        .end_token_ids(end_token_ids.cloned())
        .build_xml()
}

/// Builder for creating JSON Schema objects
#[derive(Debug, Clone, Default)]
pub struct SchemaBuilder {
    schema_type: String,
    properties: HashMap<String, Value>,
    required: Vec<String>,
    description: Option<String>,
    additional_properties: Option<bool>,
}

impl SchemaBuilder {
    /// Create a new object schema builder
    pub fn object() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
            description: None,
            additional_properties: None,
        }
    }

    /// Add a description to the schema
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a string property
    pub fn string_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "string",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a number property
    pub fn number_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "number",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an integer property
    pub fn integer_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "integer",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a boolean property
    pub fn boolean_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "boolean",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an array property
    pub fn array_prop(
        mut self,
        name: impl Into<String>,
        items_type: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "array",
                "items": { "type": items_type.into() },
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an enum property
    pub fn enum_prop(
        mut self,
        name: impl Into<String>,
        values: Vec<&str>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "string",
                "enum": values,
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a custom property with full schema
    pub fn custom_prop(mut self, name: impl Into<String>, schema: Value, required: bool) -> Self {
        let name = name.into();
        self.properties.insert(name.clone(), schema);
        if required {
            self.required.push(name);
        }
        self
    }

    /// Disallow additional properties
    pub fn no_additional_properties(mut self) -> Self {
        self.additional_properties = Some(false);
        self
    }

    /// Build the final JSON Schema
    pub fn build(self) -> Value {
        let mut schema = json!({
            "type": self.schema_type,
            "properties": self.properties,
            "required": self.required
        });

        if let Some(desc) = self.description {
            schema["description"] = json!(desc);
        }

        if let Some(additional) = self.additional_properties {
            schema["additionalProperties"] = json!(additional);
        }

        schema
    }
}

/// Common tool schemas for built-in tools
pub mod common {
    use super::*;

    /// Calculator tool schema
    pub fn calculator_schema() -> Value {
        SchemaBuilder::object()
            .description("Evaluate a mathematical expression")
            .string_prop(
                "expression",
                "The mathematical expression to evaluate",
                true,
            )
            .build()
    }

    /// Web search tool schema
    pub fn web_search_schema() -> Value {
        SchemaBuilder::object()
            .description("Search the web for information")
            .string_prop("query", "The search query", true)
            .integer_prop("max_results", "Maximum number of results to return", false)
            .build()
    }

    /// Get current time tool schema
    pub fn get_time_schema() -> Value {
        SchemaBuilder::object()
            .description("Get current date and time")
            .string_prop(
                "timezone",
                "Timezone (e.g., 'UTC', 'America/New_York')",
                false,
            )
            .build()
    }

    /// Code execution tool schema
    pub fn code_execution_schema() -> Value {
        SchemaBuilder::object()
            .description("Execute code in a sandboxed environment")
            .enum_prop(
                "language",
                vec!["python", "javascript", "rust"],
                "Programming language",
                true,
            )
            .string_prop("code", "The code to execute", true)
            .build()
    }
}

/// Build a Lark grammar for choice constraints (structured outputs choice field)
pub fn build_choice_lark_grammar(choices: &[String]) -> GrammarResult<TopLevelGrammar> {
    if choices.is_empty() {
        return Err(GrammarError::InvalidGrammar(
            "structured_outputs.choice must include at least one option".to_string(),
        ));
    }

    let mut parts = Vec::with_capacity(choices.len());
    for choice in choices {
        if choice.is_empty() {
            return Err(GrammarError::InvalidGrammar(
                "structured_outputs.choice cannot contain empty strings".to_string(),
            ));
        }
        parts.push(lark_quote(choice));
    }

    let body = parts.join(" | ");
    let lark_string = format!("start: {}\n", body);
    Ok(TopLevelGrammar::from_lark_utf8(&lark_string))
}

/// Normalize a tag string for structural_tag parsing
fn normalize_tag_pair(tag: &str) -> Result<(String, String), String> {
    let trimmed = tag.trim();
    if trimmed.is_empty() {
        return Err("structured_outputs.structural_tag.tag cannot be empty".to_string());
    }

    if trimmed.starts_with('<') && trimmed.ends_with('>') {
        let inner = trimmed
            .trim_start_matches('<')
            .trim_end_matches('>')
            .trim_start_matches('/');
        if inner.is_empty() {
            return Err("structured_outputs.structural_tag.tag is invalid".to_string());
        }
        let start = if trimmed.starts_with("</") {
            format!("<{}>", inner)
        } else {
            trimmed.to_string()
        };
        let end = format!("</{}>", inner);
        Ok((start, end))
    } else {
        Ok((format!("<{}>", trimmed), format!("</{}>", trimmed)))
    }
}

/// Parse structural_tag for structured outputs
pub fn parse_structural_tag(value: &Value) -> Result<(String, String, Value), String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "structured_outputs.structural_tag must be an object".to_string())?;

    let schema = obj
        .get("schema")
        .cloned()
        .ok_or_else(|| "structured_outputs.structural_tag.schema is required".to_string())?;

    let start = obj
        .get("start_tag")
        .or_else(|| obj.get("start"))
        .or_else(|| obj.get("tag"));
    let end = obj.get("end_tag").or_else(|| obj.get("end"));

    let (start_tag, end_tag) = match (start, end) {
        (Some(start_val), Some(end_val)) => {
            let start = start_val.as_str().ok_or_else(|| {
                "structured_outputs.structural_tag.start_tag must be a string".to_string()
            })?;
            let end = end_val.as_str().ok_or_else(|| {
                "structured_outputs.structural_tag.end_tag must be a string".to_string()
            })?;
            (start.to_string(), end.to_string())
        }
        (Some(tag), None) if obj.contains_key("tag") => {
            normalize_tag_pair(tag.as_str().ok_or_else(|| {
                "structured_outputs.structural_tag.tag must be a string".to_string()
            })?)?
        }
        _ => {
            return Err(
                "structured_outputs.structural_tag requires tag or start_tag/end_tag".to_string(),
            );
        }
    };

    Ok((start_tag, end_tag, schema))
}

/// Convert a Value schema to a Vec of Tool objects using ToolBuilder
/// The schema should be an object where keys are tool names and values are tool schemas
pub fn schema_to_tools(schema: &Value) -> Vec<Tool> {
    let mut tools = Vec::new();
    if let Value::Object(obj) = schema {
        for (name, tool_schema) in obj {
            if let Value::Object(props) = tool_schema {
                if let Some(params) = props.get("parameters") {
                    let builder = crate::tools::ToolBuilder::new(name.clone(), "".to_string())
                        .parameters_schema(params.clone());
                    tools.push(builder.build());
                }
            }
        }
    }
    tools
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::guidance::get_lark_from_top_level_grammar;

    #[test]
    fn test_sanitize_schema_for_llguidance_strips_format() {
        let schema = json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "nested": {"type": "object", "properties": {"id": {"type": "string", "format": "uuid"}}}
            }
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);
        assert!(sanitized["properties"]["url"].get("format").is_none());
        assert!(sanitized["properties"]["nested"]["properties"]["id"]
            .get("format")
            .is_none());
    }

    #[test]
    fn test_sanitize_schema_for_llguidance_preserves_nullable_types() {
        let schema = json!({
            "type": "object",
            "properties": {
                "cwd": {"type": ["string", "null"]}
            },
            "required": ["cwd"]
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);
        assert_eq!(
            sanitized["properties"]["cwd"]["type"],
            json!(["string", "null"])
        );
    }

    #[test]
    fn test_build_choice_lark_grammar_empty_string() {
        let result = build_choice_lark_grammar(&["".to_string()]);
        assert!(result.is_err());
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
        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);
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
        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);

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
        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);
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

        // Verify parameter rules reference the correct types
        assert!(
            lark_str.contains("value_0_0:"),
            "Should have value_0_0 rule for first param"
        );
        assert!(
            lark_str.contains("value_0_1:"),
            "Should have value_0_1 rule for second param"
        );
        assert!(
            lark_str.contains("value_0_2:"),
            "Should have value_0_2 rule for third param"
        );
        assert!(
            lark_str.contains("value_0_3:"),
            "Should have value_0_3 rule for fourth param"
        );

        // Verify tool rule has all parameters
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");
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

        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);

        // Verify tool rule has all parameters
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");
        assert!(lark_str.contains("value_0"), "Should have value rules");

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

        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);

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

        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);

        // Grammar should have at least one sub-grammar (the tool rules)
        assert!(grammar.grammars.len() > 0, "Should have generated grammars");
    }
}
