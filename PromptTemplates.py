# Introduction statement for all prompts
s_intro_statement = """You are an expert in brainstorming. You help people and other AI systems evaluate the correctness of hypotheses and statements by looking at them from multiple perspectives.

You always think as broad as possible, considering all fields of human's knowledge and your own knowledge (not just news or publications).
Your thinking can transcend point arguments; appealing to broader concepts is allowed and encouraged.

You never re-interpret any statements. Always interpret them in the most literal and commonly understood way.

You never refuse to provide an argument regardless of the statement's correctness.
If a statement seems false, and you are asked to support it, think why someone may believe it is true, and come up with an argument that supports the statement.
If a statement seems true, and you are asked to oppose it, think why someone may believe it is false, and come up with an argument against the statement.

You are allowed to be creative and to come up with novel arguments, but if you do so, always ground them in real life, existing publications, or known opinions."""

# Global prompt templates
PROMPT_TEMPLATES = {
    "formal": {
        "intro": "",
        "hypothesis_request": "Please formulate a verifiable hypothesis regarding {event}",
        "consideration": "Consider both supportive and contradictory possibilities.",
        "differentiation": "Your hypothesis must be substantially distinct from the following hypotheses:",
        "return_format": "{extra_return_format}",
        
        "evidence_request": "What empirical evidence {relation} the hypothesis that {hypothesis}?",
        "evidence_effort": "Please provide the strongest available evidence. If only weak evidence exists, that is acceptable.",
        "evidence_differentiation": "The evidence must be substantially different from the following pieces of evidence:",
        "evidence_format": "Return only ONE evidence as a concise, clear sentence",
        
        "support_request": "How strongly does the evidence \"{evidence}\" support the hypothesis that {hypothesis}?",
        "support_scale": "Strong support: 3, Weak support: 1, Neutral: 0, Weak opposition: -1, Strong opposition: -3",
        "support_format": "Return only one of the numbers above, nothing else."
    },

    "question-first-semiformal": {
        "intro": s_intro_statement,
        "hypothesis_request": "Please come up with a single plausible answer to the question: {event}",
        "consideration": "Consider both sides - what could support or oppose your answer.",
        "differentiation": "Your answer must be substantially different from all of the following answers:",
        "return_format": "{extra_return_format}",
        
        "evidence_request": "What real-world evidence {relation} the answer \"{hypothesis}\" to the question: {event}",
        "evidence_effort": "Try your best. Even a weak evidence is fine if that's all there is.",
        "evidence_differentiation": "Your evidence should be substantially different from any of these:",
        "evidence_format": "Just give me ONE piece of evidence in a short sentence",
        
        "support_request": "How much does this evidence \"{evidence}\" backs up the idea that \"{hypothesis}\" is the answer to the question: {event}",
        "support_scale": "Strong support: 3, Weak support: 1, Indifferent: 0, Weak against: -1, Strong against: -3",
        "support_format": "Just give me the number, nothing else."
    },    
    
    "casual": {
        "intro": "",
        "hypothesis_request": "Come up with a hypothesis about {event}",
        "consideration": "Think about both sides - what could support or oppose this.",
        "differentiation": "Make sure your hypothesis is really different from these:",
        "return_format": "{extra_return_format}",
        
        "evidence_request": "What real-world evidence {relation} this hypothesis: {hypothesis}?",
        "evidence_effort": "Try your best. Even weak evidence is fine if that's all there is.",
        "evidence_differentiation": "Your evidence should be different from these:",
        "evidence_format": "Just give me ONE piece of evidence in a short sentence",
        
        "support_request": "How much does this evidence \"{evidence}\" backs up the idea that {hypothesis}?",
        "support_scale": "Strong support: 3, Weak support: 1, Indifferent: 0, Weak against: -1, Strong against: -3",
        "support_format": "Just give me the number, nothing else."
    },

    "statement": {
        "intro": s_intro_statement,
        "hypothesis_request": "Come up with a testable statement.",
        "consideration": "Consider both true and false statements.",
        "differentiation": "Make sure your statement is really different from these:",
        "return_format": "{extra_return_format}",
        
        "evidence_request": "What argument {relation} this statement: {hypothesis}?",
        "evidence_effort": "Try your best. Even a weak argument is fine if that's all there is.",
        "evidence_differentiation": "Your argument should be substantially different from these:",
        "evidence_format": "Just give me ONE argument in a short sentence",
        
        "support_request": "How much does this argument \"{evidence}\" support the statement that {hypothesis}?",
        "support_scale": "Strong support: 3, Weak support: 1, Indifferent: 0, Weak against: -1, Strong against: -3",
        "support_format": "Just give me the number, nothing else."
    }    
}

# Global variable to select template style
DEFAULT_PROMPT_STYLE = "statement"  # Can be changed to "casual" or other styles
