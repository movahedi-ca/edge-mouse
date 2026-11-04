//! Expectation predicates for predicate-based verification (Phase 2).
//!
//! Each runbook step declares what should be true after the action.
//! [`eval`] evaluates a predicate against an [`EvalCtx`] built from live
//! AX/DOM data. `PredicateValue::Unknown` is reserved for genuinely
//! unevaluable cases (no perception data, target node gone, timeout).
//! Unknown is never a confirmation and never a denial: the verifier maps
//! it to `Inconclusive`, and the runner escalates instead of clicking.

/// A declared expectation about the page state after an action.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    /// The target node is enabled and visible.
    EnabledVisible,
    /// Some node carries this text.
    HasText(String),
    /// Some node carries this attribute value.
    HasAttribute { name: String, value: String },
    /// Some node has this accessibility role.
    RoleIs(String),
    /// The target input's value equals this string.
    ValueEquals(String),
    /// A node attribute matches this pattern (syntax in [`match_pattern`]).
    AttributeMatches { name: String, pattern: String },
    /// The target node has keyboard focus.
    Focused,
}

/// The evaluated value of a predicate.
#[derive(Debug, Clone, PartialEq)]
pub enum PredicateValue {
    /// A boolean-valued predicate resolved to true/false.
    Bool(bool),
    /// A text-valued predicate resolved to this string.
    Text(String),
    /// Could not be determined (perception empty, node gone, timeout).
    Unknown,
}

impl PredicateValue {
    /// True only for `Bool(true)`. `Text` and `Unknown` are never
    /// confirmation: the verifier treats them as non-true.
    pub fn is_true(&self) -> bool {
        matches!(self, PredicateValue::Bool(true))
    }
}

/// One accessible node, as seen by the verifier. Built from the AX/DOM
/// snapshot by the perception layer; the verifier never parses the DOM
/// itself.
#[derive(Debug, Clone, Default)]
pub struct AxNode {
    pub node_id: u64,
    pub role: String,
    pub name: String,
    pub text: String,
    pub value: String,
    pub enabled: bool,
    pub visible: bool,
    pub focused: bool,
    /// `input[type=password]`: its box is zeroed before any screenshot
    /// bytes leave the VM.
    pub is_password: bool,
    /// Device-pixel box as it appears in captures: (x, y, w, h).
    pub box_px: (u32, u32, u32, u32),
    pub attributes: Vec<(String, String)>,
}

/// AX/DOM context a predicate is evaluated against.
#[derive(Debug, Clone, Default)]
pub struct EvalCtx {
    pub nodes: Vec<AxNode>,
    pub url: String,
    /// The step's target node. Several predicates are target-relative and
    /// are `Unknown` without it.
    pub target_node_id: Option<u64>,
}

impl EvalCtx {
    /// The step's target node, if the id resolves against this snapshot.
    pub fn target(&self) -> Option<&AxNode> {
        self.target_node_id
            .and_then(|id| self.nodes.iter().find(|n| n.node_id == id))
    }

    /// Password-field boxes for screenshot redaction, in device px.
    pub fn password_boxes(&self) -> Vec<(u32, u32, u32, u32)> {
        self.nodes
            .iter()
            .filter(|n| n.is_password)
            .map(|n| n.box_px)
            .collect()
    }
}

/// Evaluate a predicate against live AX/DOM context.
///
/// Returns `Unknown` when the context cannot answer: empty perception,
/// the target id missing from the snapshot, or the queried attribute
/// absent where absence is not evidence. Callers must never treat
/// `Unknown` as confirmation.
pub fn eval(predicate: &Predicate, ctx: &EvalCtx) -> PredicateValue {
    // Empty perception answers nothing honestly.
    if ctx.nodes.is_empty() {
        return PredicateValue::Unknown;
    }
    match predicate {
        Predicate::EnabledVisible => match ctx.target() {
            Some(n) => PredicateValue::Bool(n.enabled && n.visible),
            None => PredicateValue::Unknown,
        },
        Predicate::HasText(want) => {
            let hit = ctx
                .nodes
                .iter()
                .any(|n| n.name.contains(want) || n.text.contains(want));
            PredicateValue::Bool(hit)
        }
        Predicate::HasAttribute { name, value } => {
            let hit = ctx
                .nodes
                .iter()
                .any(|n| n.attributes.iter().any(|(k, v)| k == name && v == value));
            PredicateValue::Bool(hit)
        }
        Predicate::RoleIs(role) => match ctx.target() {
            Some(n) => PredicateValue::Bool(n.role.eq_ignore_ascii_case(role)),
            None => PredicateValue::Unknown,
        },
        Predicate::ValueEquals(want) => match ctx.target() {
            Some(n) => PredicateValue::Bool(n.value == *want),
            None => PredicateValue::Unknown,
        },
        Predicate::AttributeMatches { name, pattern } => match ctx.target() {
            Some(n) => match n.attributes.iter().find(|(k, _)| k == name) {
                Some((_, v)) => PredicateValue::Bool(match_pattern(v, pattern)),
                // Attribute absent on a present node: the pattern does not
                // match. (A missing *node* is Unknown; a present node with
                // a missing attribute is a resolved false.)
                None => PredicateValue::Bool(false),
            },
            None => PredicateValue::Unknown,
        },
        Predicate::Focused => match ctx.target() {
            Some(n) => PredicateValue::Bool(n.focused),
            None => PredicateValue::Unknown,
        },
    }
}

/// Pattern syntax for [`Predicate::AttributeMatches`]:
/// `prefix:<s>` for starts-with, `contains:<s>` for substring,
/// `exact:<s>` or a bare `<s>` for equality.
pub fn match_pattern(value: &str, pattern: &str) -> bool {
    if let Some(rest) = pattern.strip_prefix("prefix:") {
        return value.starts_with(rest);
    }
    if let Some(rest) = pattern.strip_prefix("contains:") {
        return value.contains(rest);
    }
    let rest = pattern.strip_prefix("exact:").unwrap_or(pattern);
    value == rest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64) -> AxNode {
        AxNode {
            node_id: id,
            role: "button".to_string(),
            name: "Submit".to_string(),
            text: "Submit".to_string(),
            value: String::new(),
            enabled: true,
            visible: true,
            focused: false,
            is_password: false,
            box_px: (0, 0, 10, 10),
            attributes: vec![("aria-expanded".to_string(), "true".to_string())],
        }
    }

    fn ctx_with_target() -> EvalCtx {
        EvalCtx {
            nodes: vec![node(7)],
            url: "https://portal.example/form".to_string(),
            target_node_id: Some(7),
        }
    }

    #[test]
    fn eval_unknown_for_every_variant_on_empty_context() {
        let ctx = EvalCtx::default();
        let preds = [
            Predicate::EnabledVisible,
            Predicate::HasText("Submit".to_string()),
            Predicate::HasAttribute {
                name: "aria-expanded".to_string(),
                value: "true".to_string(),
            },
            Predicate::RoleIs("button".to_string()),
            Predicate::ValueEquals("mohammad".to_string()),
            Predicate::AttributeMatches {
                name: "href".to_string(),
                pattern: "/settings".to_string(),
            },
            Predicate::Focused,
        ];
        for p in &preds {
            assert_eq!(eval(p, &ctx), PredicateValue::Unknown, "{p:?}");
        }
    }

    #[test]
    fn eval_target_relative_predicates() {
        let ctx = ctx_with_target();
        assert_eq!(
            eval(&Predicate::EnabledVisible, &ctx),
            PredicateValue::Bool(true)
        );
        assert_eq!(
            eval(&Predicate::RoleIs("button".to_string()), &ctx),
            PredicateValue::Bool(true)
        );
        assert_eq!(
            eval(&Predicate::RoleIs("link".to_string()), &ctx),
            PredicateValue::Bool(false)
        );
        assert_eq!(eval(&Predicate::Focused, &ctx), PredicateValue::Bool(false));

        // Target id missing from the snapshot: Unknown, not false.
        let mut gone = ctx.clone();
        gone.target_node_id = Some(999);
        assert_eq!(
            eval(&Predicate::EnabledVisible, &gone),
            PredicateValue::Unknown
        );
        assert_eq!(
            eval(&Predicate::ValueEquals("x".to_string()), &gone),
            PredicateValue::Unknown
        );
    }

    #[test]
    fn eval_text_and_attribute_predicates() {
        let ctx = ctx_with_target();
        assert_eq!(
            eval(&Predicate::HasText("Submit".to_string()), &ctx),
            PredicateValue::Bool(true)
        );
        assert_eq!(
            eval(&Predicate::HasText("Done".to_string()), &ctx),
            PredicateValue::Bool(false)
        );
        assert_eq!(
            eval(
                &Predicate::HasAttribute {
                    name: "aria-expanded".to_string(),
                    value: "true".to_string(),
                },
                &ctx
            ),
            PredicateValue::Bool(true)
        );
        assert_eq!(
            eval(
                &Predicate::HasAttribute {
                    name: "aria-expanded".to_string(),
                    value: "false".to_string(),
                },
                &ctx
            ),
            PredicateValue::Bool(false)
        );
    }

    #[test]
    fn eval_attribute_matches_syntax() {
        let ctx = ctx_with_target();
        let pat = |p: &str| Predicate::AttributeMatches {
            name: "aria-expanded".to_string(),
            pattern: p.to_string(),
        };
        assert_eq!(eval(&pat("prefix:tr"), &ctx), PredicateValue::Bool(true));
        assert_eq!(eval(&pat("prefix:fa"), &ctx), PredicateValue::Bool(false));
        assert_eq!(eval(&pat("contains:ru"), &ctx), PredicateValue::Bool(true));
        assert_eq!(eval(&pat("exact:true"), &ctx), PredicateValue::Bool(true));
        assert_eq!(eval(&pat("true"), &ctx), PredicateValue::Bool(true));
        assert_eq!(eval(&pat("false"), &ctx), PredicateValue::Bool(false));
        // Attribute absent on a present node: resolved false, not Unknown.
        let missing = Predicate::AttributeMatches {
            name: "href".to_string(),
            pattern: "x".to_string(),
        };
        assert_eq!(eval(&missing, &ctx), PredicateValue::Bool(false));
    }

    #[test]
    fn eval_value_equals() {
        let mut ctx = ctx_with_target();
        ctx.nodes[0].value = "mohammad".to_string();
        assert_eq!(
            eval(&Predicate::ValueEquals("mohammad".to_string()), &ctx),
            PredicateValue::Bool(true)
        );
        assert_eq!(
            eval(&Predicate::ValueEquals("other".to_string()), &ctx),
            PredicateValue::Bool(false)
        );
    }

    #[test]
    fn is_true_only_for_bool_true() {
        assert!(PredicateValue::Bool(true).is_true());
        assert!(!PredicateValue::Bool(false).is_true());
        assert!(!PredicateValue::Text("x".to_string()).is_true());
        assert!(!PredicateValue::Unknown.is_true());
    }

    #[test]
    fn password_boxes_collects_only_password_nodes() {
        let mut ctx = ctx_with_target();
        ctx.nodes[0].is_password = true;
        ctx.nodes[0].box_px = (10, 20, 100, 24);
        ctx.nodes.push(node(8));
        assert_eq!(ctx.password_boxes(), vec![(10, 20, 100, 24)]);
    }
}
