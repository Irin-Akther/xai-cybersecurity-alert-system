"""
User Profiler — multi-persona, reading-level-adaptive user classification.

Literacy levels (technical detail depth):
  HOME   — non-technical; plain everyday language, no jargon.
  SMB    — IT-comfortable or business-minded; accessible technical language.
  ADMIN  — security professional; full forensic and regulatory detail.

Personas (tone and context) map onto those three levels:
  HOME  → Kid, Teenager, Housewife, Cashier, General Employee
  SMB   → Business Owner, Student, Executive / Manager
  ADMIN → Compliance / Auditor, Security Analyst (original ADMIN)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LiteracyLevel(str, Enum):
    HOME = "HOME"
    SMB = "SMB"
    ADMIN = "ADMIN"


class Persona(str, Enum):
    # HOME-level personas
    KID             = "Kid"
    TEENAGER        = "Teenager"
    HOUSEWIFE       = "Housewife"
    CASHIER         = "Cashier"
    GENERAL_EMPLOYEE = "General Employee"
    # SMB-level personas
    BUSINESS_OWNER  = "Business Owner"
    STUDENT         = "Student"
    EXECUTIVE       = "Executive / Manager"
    # ADMIN-level personas
    COMPLIANCE      = "Compliance / Auditor"
    SECURITY_ANALYST = "Security Analyst"


# Persona → LiteracyLevel mapping
PERSONA_LEVEL: dict[Persona, LiteracyLevel] = {
    Persona.KID:              LiteracyLevel.HOME,
    Persona.TEENAGER:         LiteracyLevel.HOME,
    Persona.HOUSEWIFE:        LiteracyLevel.HOME,
    Persona.CASHIER:          LiteracyLevel.HOME,
    Persona.GENERAL_EMPLOYEE: LiteracyLevel.HOME,
    Persona.BUSINESS_OWNER:   LiteracyLevel.SMB,
    Persona.STUDENT:          LiteracyLevel.SMB,
    Persona.EXECUTIVE:        LiteracyLevel.SMB,
    Persona.COMPLIANCE:       LiteracyLevel.ADMIN,
    Persona.SECURITY_ANALYST: LiteracyLevel.ADMIN,
}

# Tone instructions for NLG, per persona
PERSONA_TONE: dict[Persona, str] = {
    Persona.KID: (
        "Use very simple words a 10-year-old would understand. "
        "Use a friendly, encouraging tone. Compare the threat to something from everyday life "
        "like a stranger at the door or a locked toy box. Keep it under 50 words."
    ),
    Persona.TEENAGER: (
        "Use casual, relatable language for a teenager. "
        "You can use light analogies from gaming or social media. "
        "Be direct and practical. Keep it under 70 words."
    ),
    Persona.HOUSEWIFE: (
        "Explain using simple home and family analogies — like a burglar alarm or a locked front door. "
        "Avoid all technical terms. Focus on what action to take to keep the family safe. Under 70 words."
    ),
    Persona.CASHIER: (
        "Use very simple workplace language. Explain what happened and one clear thing the person should do. "
        "No technical jargon whatsoever. Under 60 words."
    ),
    Persona.GENERAL_EMPLOYEE: (
        "Explain in plain office language. Focus on whether this affects their work device or account "
        "and what they should do right now. Keep it practical and under 80 words."
    ),
    Persona.BUSINESS_OWNER: (
        "Focus on business impact: what is at risk (data, money, operations), "
        "how serious it is, and what the immediate business decision should be. "
        "Avoid deep technical detail. Under 100 words."
    ),
    Persona.STUDENT: (
        "Explain clearly and include a brief learning point — what technique was used and why "
        "the model flagged it. This person is studying cybersecurity so some technical context is helpful. "
        "Under 120 words."
    ),
    Persona.EXECUTIVE: (
        "Executive summary style. Lead with risk level and business impact. "
        "One sentence on cause, one on recommended action. No technical jargon. Under 80 words."
    ),
    Persona.COMPLIANCE: (
        "Frame the alert in compliance and regulatory terms. Reference relevant standards "
        "(NIST, ISO 27001, GDPR, HIPAA as applicable). Include audit trail implications "
        "and required notification steps. Up to 150 words."
    ),
    Persona.SECURITY_ANALYST: (
        "Provide full technical analysis for a SOC analyst. Include top SHAP features, "
        "likely attack vector, MITRE ATT&CK mapping, and immediate containment steps. Up to 200 words."
    ),
}


@dataclass
class UserProfile:
    level: LiteracyLevel
    persona: Persona
    display_name: str = "User"
    organisation: str = ""
    preferred_language: str = "en"
    extra_context: dict = field(default_factory=dict)

    @property
    def reading_grade(self) -> str:
        mapping = {
            LiteracyLevel.HOME: "simple everyday language",
            LiteracyLevel.SMB:  "technical but accessible",
            LiteracyLevel.ADMIN: "full technical / regulatory detail",
        }
        return mapping[self.level]

    @property
    def detail_depth(self) -> str:
        mapping = {
            LiteracyLevel.HOME:  "brief",
            LiteracyLevel.SMB:   "standard",
            LiteracyLevel.ADMIN: "detailed",
        }
        return mapping[self.level]

    @property
    def tone_instruction(self) -> str:
        return PERSONA_TONE[self.persona]


def make_profile(persona: Persona, display_name: str = "", organisation: str = "") -> UserProfile:
    """Convenience constructor — derive literacy level from persona automatically."""
    level = PERSONA_LEVEL[persona]
    return UserProfile(
        level=level,
        persona=persona,
        display_name=display_name or persona.value,
        organisation=organisation,
    )


# Pre-built profiles for quick use
HOME_PROFILE     = make_profile(Persona.GENERAL_EMPLOYEE, "Home User")
SMB_PROFILE      = make_profile(Persona.BUSINESS_OWNER,   "IT Staff")
ADMIN_PROFILE    = make_profile(Persona.SECURITY_ANALYST, "Security Analyst")

KID_PROFILE      = make_profile(Persona.KID)
TEEN_PROFILE     = make_profile(Persona.TEENAGER)
HOUSEWIFE_PROFILE = make_profile(Persona.HOUSEWIFE)
CASHIER_PROFILE  = make_profile(Persona.CASHIER)
EMPLOYEE_PROFILE = make_profile(Persona.GENERAL_EMPLOYEE)
OWNER_PROFILE    = make_profile(Persona.BUSINESS_OWNER)
STUDENT_PROFILE  = make_profile(Persona.STUDENT)
EXEC_PROFILE     = make_profile(Persona.EXECUTIVE)
COMPLIANCE_PROFILE = make_profile(Persona.COMPLIANCE)


# ---------------------------------------------------------------------------
# Questionnaire
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "id": "q1",
        "text": "How would you describe yourself?",
        "options": {
            "a": ("A child or young student (under 13)", 0),
            "b": ("A teenager or high school student", 1),
            "c": ("A home user, housewife, or non-office worker", 2),
            "d": ("An office or business professional", 3),
        },
    },
    {
        "id": "q2",
        "text": "How familiar are you with computer security terms like 'firewall' or 'malware'?",
        "options": {
            "a": ("I don't know what these mean", 0),
            "b": ("I've heard of them but don't fully understand them", 1),
            "c": ("I understand the basics", 2),
            "d": ("I use these terms regularly in my work", 3),
        },
    },
    {
        "id": "q3",
        "text": "What is your job role?",
        "options": {
            "a": ("Student / no job", 0),
            "b": ("Cashier, retail, or service worker", 1),
            "c": ("Business owner, manager, or executive", 2),
            "d": ("IT, security, or compliance professional", 3),
        },
    },
    {
        "id": "q4",
        "text": "When you get a security alert, what matters most to you?",
        "options": {
            "a": ("Just tell me if I'm safe and what to do", 0),
            "b": ("Tell me what's at risk for my business or work", 1),
            "c": ("Give me the technical details and fix steps", 2),
            "d": ("Show me the compliance and audit implications", 3),
        },
    },
]


def _score_to_persona(answers: dict[str, str]) -> Persona:
    score = 0
    for q in QUESTIONS:
        sel = answers.get(q["id"])
        if sel and sel in q["options"]:
            score += q["options"][sel][1]

    if score <= 2:
        return Persona.KID
    elif score <= 4:
        return Persona.TEENAGER
    elif score <= 5:
        return Persona.HOUSEWIFE
    elif score <= 6:
        return Persona.GENERAL_EMPLOYEE
    elif score <= 7:
        return Persona.BUSINESS_OWNER
    elif score <= 8:
        return Persona.STUDENT
    elif score <= 9:
        return Persona.EXECUTIVE
    elif score <= 10:
        return Persona.COMPLIANCE
    else:
        return Persona.SECURITY_ANALYST


class UserProfiler:
    """Determines a user's cybersecurity persona via questionnaire or direct assignment."""

    def from_answers(self, answers: dict[str, str], display_name: str = "User") -> UserProfile:
        persona = _score_to_persona(answers)
        return make_profile(persona, display_name=display_name)

    def from_persona(
        self,
        persona: str | Persona,
        display_name: str = "",
        organisation: str = "",
    ) -> UserProfile:
        if isinstance(persona, str):
            persona = Persona(persona)
        return make_profile(persona, display_name=display_name, organisation=organisation)

    def from_level(
        self,
        level: str | LiteracyLevel,
        display_name: str = "User",
        organisation: str = "",
    ) -> UserProfile:
        """Legacy helper — maps bare level to default persona for that level."""
        if isinstance(level, str):
            level = LiteracyLevel(level.upper())
        defaults = {
            LiteracyLevel.HOME:  Persona.GENERAL_EMPLOYEE,
            LiteracyLevel.SMB:   Persona.BUSINESS_OWNER,
            LiteracyLevel.ADMIN: Persona.SECURITY_ANALYST,
        }
        return make_profile(defaults[level], display_name=display_name, organisation=organisation)

    def interactive_questionnaire(self) -> UserProfile:
        print("\n=== Cybersecurity Literacy Assessment ===\n")
        answers: dict[str, str] = {}
        for q in QUESTIONS:
            print(q["text"])
            for key, (label, _) in q["options"].items():
                print(f"  [{key}] {label}")
            while True:
                choice = input("Your choice: ").strip().lower()
                if choice in q["options"]:
                    answers[q["id"]] = choice
                    break
                print("    Please enter a valid option.")
            print()
        name = input("Your name or alias (optional): ").strip() or "User"
        profile = self.from_answers(answers, display_name=name)
        print(f"\nProfile: {profile.persona.value} ({profile.level.value})\n")
        return profile

    @staticmethod
    def get_questions() -> list[dict]:
        return QUESTIONS

    @staticmethod
    def all_personas() -> list[Persona]:
        return list(Persona)
