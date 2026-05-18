"""
User Profiler — 3-level literacy classifier for reading-level-adaptive alert explanations.

Literacy levels:
  HOME   — non-technical home user; needs plain-English, jargon-free explanations.
  SMB    — small/medium business IT staff; comfortable with networking basics.
  ADMIN  — security professional / SOC analyst; wants full technical detail.

Profile is determined via a short self-assessment questionnaire or can be set
programmatically for API / batch use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LiteracyLevel(str, Enum):
    HOME = "HOME"
    SMB = "SMB"
    ADMIN = "ADMIN"


@dataclass
class UserProfile:
    level: LiteracyLevel
    display_name: str = "User"
    organisation: str = ""
    preferred_language: str = "en"
    extra_context: dict = field(default_factory=dict)

    @property
    def reading_grade(self) -> str:
        """Approximate reading-grade target for NLG prompts."""
        mapping = {
            LiteracyLevel.HOME: "grade 6 (simple, everyday language)",
            LiteracyLevel.SMB: "grade 10 (technical but accessible)",
            LiteracyLevel.ADMIN: "graduate-level (full technical detail, include IoCs and MITRE ATT&CK references)",
        }
        return mapping[self.level]

    @property
    def detail_depth(self) -> str:
        """Instruction depth hint for NLG module."""
        mapping = {
            LiteracyLevel.HOME: "brief",
            LiteracyLevel.SMB: "standard",
            LiteracyLevel.ADMIN: "detailed",
        }
        return mapping[self.level]


# ---------------------------------------------------------------------------
# Pre-built profiles for quick use
# ---------------------------------------------------------------------------
HOME_PROFILE = UserProfile(level=LiteracyLevel.HOME, display_name="Home User")
SMB_PROFILE = UserProfile(level=LiteracyLevel.SMB, display_name="IT Staff")
ADMIN_PROFILE = UserProfile(level=LiteracyLevel.ADMIN, display_name="Security Analyst")


# ---------------------------------------------------------------------------
# Questionnaire-based profiler
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "id": "q1",
        "text": "How would you describe your role?",
        "options": {
            "a": ("Home user / personal device owner", 0),
            "b": ("IT support staff or network administrator", 1),
            "c": ("Security engineer, SOC analyst, or penetration tester", 2),
        },
    },
    {
        "id": "q2",
        "text": "How familiar are you with terms like 'SYN flood', 'DDoS', or 'port scan'?",
        "options": {
            "a": ("I don't know what these mean", 0),
            "b": ("I've heard of them and understand the basics", 1),
            "c": ("I work with these concepts regularly", 2),
        },
    },
    {
        "id": "q3",
        "text": "When you see a security alert, what do you want most?",
        "options": {
            "a": ("A simple explanation of whether I'm in danger and what to do", 0),
            "b": ("Which system or service is affected and a suggested fix", 1),
            "c": ("Full technical details including traffic features and MITRE ATT&CK mapping", 2),
        },
    },
    {
        "id": "q4",
        "text": "Have you ever reviewed firewall logs, PCAP files, or IDS alerts?",
        "options": {
            "a": ("No", 0),
            "b": ("Occasionally, with guidance", 1),
            "c": ("Yes, regularly as part of my job", 2),
        },
    },
]


def score_to_level(total_score: int) -> LiteracyLevel:
    """Map questionnaire score to a literacy level."""
    if total_score <= 2:
        return LiteracyLevel.HOME
    elif total_score <= 5:
        return LiteracyLevel.SMB
    else:
        return LiteracyLevel.ADMIN


class UserProfiler:
    """Determines a user's cybersecurity literacy via questionnaire or direct assignment."""

    def from_answers(self, answers: dict[str, str], display_name: str = "User") -> UserProfile:
        """Build a profile from questionnaire answers.

        Args:
            answers: Dict mapping question id to selected option key, e.g. {"q1": "b", ...}
            display_name: User's name or alias.

        Returns:
            A UserProfile with the inferred literacy level.
        """
        score = 0
        for q in QUESTIONS:
            qid = q["id"]
            selected = answers.get(qid)
            if selected and selected in q["options"]:
                score += q["options"][selected][1]

        level = score_to_level(score)
        return UserProfile(level=level, display_name=display_name)

    def from_level(
        self,
        level: str | LiteracyLevel,
        display_name: str = "User",
        organisation: str = "",
    ) -> UserProfile:
        """Directly construct a profile from a known level string."""
        if isinstance(level, str):
            level = LiteracyLevel(level.upper())
        return UserProfile(level=level, display_name=display_name, organisation=organisation)

    def interactive_questionnaire(self) -> UserProfile:
        """Run an interactive CLI questionnaire and return the resulting profile."""
        print("\n=== Cybersecurity Literacy Assessment ===")
        print("Answer a few quick questions so we can tailor alerts to your expertise.\n")

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
                print("    Please enter a, b, or c.")
            print()

        name = input("Your name or alias (optional, press Enter to skip): ").strip() or "User"
        profile = self.from_answers(answers, display_name=name)
        print(f"\nProfile: {profile.level.value} — {profile.reading_grade}\n")
        return profile

    @staticmethod
    def get_questions() -> list[dict]:
        """Return the questionnaire definition for use in a web UI."""
        return QUESTIONS
