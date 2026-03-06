#!/usr/bin/env python3
"""
Seniority Level Detection System

Analyzes resume experiences and skills to determine professional seniority levels
(junior/mid-level/senior/principal) based on multiple factors including experience
duration, skill depth, leadership indicators, and achievement complexity.
"""

import re
from typing import List, Dict, Set, Optional
from datetime import datetime
import json


class SeniorityDetector:
    """
    Detects professional seniority levels based on resume analysis.
    Uses multiple signals including experience duration, skill mastery,
    leadership indicators, and achievement complexity.
    """

    # Seniority level definitions
    SENIORITY_LEVELS = {
        'junior': {
            'min_years': 0,
            'max_years': 2,
            'key_indicators': ['learning', 'training', 'assisting', 'supporting'],
            'skill_depth': 'basic',
            'typical_titles': ['junior', 'associate', 'trainee', 'intern']
        },
        'mid-level': {
            'min_years': 2,
            'max_years': 5,
            'key_indicators': ['developing', 'implementing', 'maintaining', 'improving'],
            'skill_depth': 'intermediate',
            'typical_titles': ['developer', 'analyst', 'specialist', 'engineer']
        },
        'senior': {
            'min_years': 5,
            'max_years': 10,
            'key_indicators': ['leading', 'architecting', 'mentoring', 'strategizing'],
            'skill_depth': 'advanced',
            'typical_titles': ['senior', 'lead', 'principal', 'architect']
        },
        'principal': {
            'min_years': 10,
            'key_indicators': ['executing', 'directing', 'innovating', 'transforming'],
            'skill_depth': 'expert',
            'typical_titles': ['principal', 'director', 'vp', 'chief', 'head']
        }
    }

    # Leadership indicators (expanded to catch more patterns)
    LEADERSHIP_KEYWORDS = {
        'team_leadership': [
            'led team', 'managed team', 'supervised', 'directed', 'oversaw',
            'team lead', 'leading team', 'led a team', 'managed a team',
            'head of', 'led development', 'led engineering',
            'led the', 'managed the', 'leading the'
        ],
        'project_leadership': [
            'project lead', 'project manager', 'coordinated', 'orchestrated',
            'spearheaded', 'drove', 'championed', 'owned',
            'led the project', 'managing project', 'led project'
        ],
        'mentorship': [
            'mentored', 'coached', 'trained', 'guided', 'onboarded',
            'mentoring', 'coaching', 'training', 'guiding',
            'helped junior', 'supported junior', 'grew the team'
        ],
        'strategic': [
            'strategic', 'vision', 'roadmap', 'planning', 'initiated',
            'architecture', 'designed system', 'designed the'
        ]
    }

    # Technical achievement complexity indicators
    TECHNICAL_COMPLEXITY = {
        'scale_indicators': ['million', 'billion', 'thousand', 'users', 'requests', 'transactions'],
        'impact_indicators': ['revenue', 'savings', 'efficiency', 'performance', 'reduction'],
        'innovation_indicators': ['architected', 'designed', 'invented', 'patented', 'novel']
    }

    def __init__(self):
        """Initialize the seniority detector with default configurations."""
        self.experience_parser = ExperienceParser()
    
    def _get_exp_text(self, exp: Dict) -> str:
        """
        Get text content from experience, handling both old and new field names.
        
        Old style: title, description
        New style: title_line, body, snippet, raw
        """
        # Get title (try multiple field names)
        title = exp.get('title') or exp.get('title_line') or exp.get('role') or ''
        
        # Get description (try multiple field names)
        description = (
            exp.get('description') or 
            exp.get('body') or 
            exp.get('snippet') or 
            exp.get('raw') or 
            ''
        )
        
        return f"{title} {description}".lower()

    def detect_seniority(self, experiences: List[Dict], skills: Set[str],
                        education: Optional[List[Dict]] = None) -> Dict[str, any]:
        """
        Main method to detect seniority level from resume data.

        Args:
            experiences: List of work experiences with title, duration, description
            skills: Set of identified skills
            education: Optional education history

        Returns:
            Dictionary containing seniority level, confidence score, and reasoning
        """
        # Calculate total years of experience
        total_years = self._calculate_total_experience(experiences)

        # Analyze experience quality and complexity
        experience_quality = self._analyze_experience_quality(experiences)

        # Detect leadership indicators
        leadership_score = self._detect_leadership_indicators(experiences)

        # Assess technical skill depth
        skill_depth_score = self._assess_skill_depth(skills, experiences)

        # Analyze achievement complexity
        achievement_complexity = self._analyze_achievement_complexity(experiences)

        # Determine seniority level based on combined signals
        seniority_level = self._determine_seniority_level(
            total_years=total_years,
            experience_quality=experience_quality,
            leadership_score=leadership_score,
            skill_depth_score=skill_depth_score,
            achievement_complexity=achievement_complexity
        )

        # Calculate confidence score
        confidence = self._calculate_confidence(
            total_years=total_years,
            data_completeness=self._assess_data_completeness(experiences, skills),
            signal_consistency=self._assess_signal_consistency(
                seniority_level, total_years, leadership_score, skill_depth_score
            )
        )

        return {
            'level': seniority_level,
            'confidence': confidence,
            'total_years_experience': total_years,
            'experience_quality_score': experience_quality['overall_score'],
            'leadership_score': leadership_score,
            'skill_depth_score': skill_depth_score,
            'achievement_complexity_score': achievement_complexity['overall_score'],
            'reasoning': self._generate_reasoning(
                seniority_level, total_years, experience_quality, leadership_score,
                skill_depth_score, achievement_complexity
            ),
            'recommendations': self._generate_recommendations(
                seniority_level, total_years, skill_depth_score, leadership_score
            )
        }

    def _calculate_total_experience(self, experiences: List[Dict]) -> float:
        """Calculate total years of professional experience."""
        total_months = 0

        for exp in experiences:
            duration = exp.get('duration', '')
            if not duration:
                # Try to extract from title_line (e.g., "TechCorp | Software Engineer | Jan 2020 - Present")
                title_line = exp.get('title_line') or exp.get('title') or ''
                duration = self.experience_parser.extract_duration_from_text(title_line)
            
            if not duration:
                # Try to extract from raw/body/snippet/description
                text_content = self._get_exp_text(exp)
                duration = self.experience_parser.extract_duration_from_text(text_content)

            months = self.experience_parser.parse_duration_to_months(duration)
            total_months += months

        return round(total_months / 12, 1)

    def _analyze_experience_quality(self, experiences: List[Dict]) -> Dict:
        """Analyze the quality and complexity of experiences."""
        quality_scores = []
        complexity_indicators = []

        for exp in experiences:
            description = self._get_exp_text(exp)
            
            # Get title for seniority detection (try multiple field names)
            title_raw = exp.get('title') or exp.get('title_line') or exp.get('role') or ''

            # Check for seniority indicators in titles
            title_seniority = self._detect_title_seniority(title_raw)

            # Analyze description complexity
            desc_complexity = self._analyze_description_complexity(description)

            # Check for achievement metrics
            has_metrics = self._detect_achievement_metrics(description)

            # Check for technical depth
            technical_depth = self._assess_technical_depth(description)

            quality_score = (title_seniority * 0.3 +
                           desc_complexity * 0.25 +
                           has_metrics * 0.25 +
                           technical_depth * 0.2)

            quality_scores.append(quality_score)

            complexity_indicators.append({
                'title_seniority': title_seniority,
                'description_complexity': desc_complexity,
                'has_metrics': has_metrics,
                'technical_depth': technical_depth
            })

        return {
            'overall_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'individual_scores': quality_scores,
            'complexity_indicators': complexity_indicators
        }

    def _detect_leadership_indicators(self, experiences: List[Dict]) -> float:
        """Detect leadership experience and responsibilities."""
        leadership_signals = {
            'team_leadership': 0,
            'project_leadership': 0,
            'mentorship': 0,
            'strategic_contributions': 0
        }

        for exp in experiences:
            description = self._get_exp_text(exp)

            # Check for team leadership
            for indicator in self.LEADERSHIP_KEYWORDS['team_leadership']:
                if indicator in description:
                    leadership_signals['team_leadership'] += 1

            # Check for project leadership
            for indicator in self.LEADERSHIP_KEYWORDS['project_leadership']:
                if indicator in description:
                    leadership_signals['project_leadership'] += 1

            # Check for mentorship
            for indicator in self.LEADERSHIP_KEYWORDS['mentorship']:
                if indicator in description:
                    leadership_signals['mentorship'] += 1

            # Check for strategic contributions
            for indicator in self.LEADERSHIP_KEYWORDS['strategic']:
                if indicator in description:
                    leadership_signals['strategic_contributions'] += 1

        # Calculate weighted leadership score
        weights = {'team_leadership': 0.3, 'project_leadership': 0.25,
                  'mentorship': 0.25, 'strategic_contributions': 0.2}

        total_score = sum(leadership_signals[k] * weights[k]
                         for k in leadership_signals)

        # Normalize to 0-1 scale
        return min(total_score / 10.0, 1.0)

    def _assess_skill_depth(self, skills: Set[str], experiences: List[Dict]) -> float:
        """Assess the depth and mastery of technical skills."""
        if not skills:
            return 0.0

        # Count advanced skills (frameworks, architectures, etc.)
        advanced_skills = {'architecture', 'design', 'lead', 'principal', 'distributed', 'scalability'}
        intermediate_skills = {'framework', 'library', 'api', 'database', 'cloud'}

        advanced_count = sum(1 for skill in skills if any(adv in skill.lower() for adv in advanced_skills))
        intermediate_count = sum(1 for skill in skills if any(inter in skill.lower() for inter in intermediate_skills))
        basic_count = len(skills) - advanced_count - intermediate_count

        # Calculate weighted skill depth score
        total_score = (advanced_count * 1.0 + intermediate_count * 0.6 + basic_count * 0.3)
        max_possible = len(skills) * 1.0

        return total_score / max_possible if max_possible > 0 else 0.0

    def _analyze_achievement_complexity(self, experiences: List[Dict]) -> Dict:
        """Analyze the complexity and impact of achievements."""
        complexity_scores = []

        for exp in experiences:
            description = self._get_exp_text(exp)
            score = 0

            # Check for scale indicators (millions, billions, etc.)
            for indicator in self.TECHNICAL_COMPLEXITY['scale_indicators']:
                if indicator in description:
                    # Extract numbers and calculate impact
                    numbers = self._extract_numbers(description)
                    if numbers:
                        max_number = max(numbers)
                        if max_number >= 1000000:
                            score += 1.0
                        elif max_number >= 1000:
                            score += 0.6
                        else:
                            score += 0.3

            # Check for business impact
            for indicator in self.TECHNICAL_COMPLEXITY['impact_indicators']:
                if indicator in description:
                    score += 0.8

            # Check for innovation
            for indicator in self.TECHNICAL_COMPLEXITY['innovation_indicators']:
                if indicator in description:
                    score += 1.0

            complexity_scores.append(min(score, 3.0))  # Cap at 3.0

        return {
            'overall_score': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            'individual_scores': complexity_scores
        }

    def _determine_seniority_level(self, total_years: float, experience_quality: Dict,
                                 leadership_score: float, skill_depth_score: float,
                                 achievement_complexity: Dict) -> str:
        """Determine the seniority level based on all signals."""
        # Calculate weighted composite score
        weights = {
            'years_experience': 0.35,  # Increased weight for years
            'experience_quality': 0.2,
            'leadership': 0.2,  # Reduced from 0.25
            'skill_depth': 0.15,
            'achievement_complexity': 0.1  # Reduced from 0.15
        }

        composite_score = (
            min(total_years / 12.0, 1.0) * weights['years_experience'] +  # Scaled to 12 years instead of 15
            experience_quality['overall_score'] * weights['experience_quality'] +
            leadership_score * weights['leadership'] +
            skill_depth_score * weights['skill_depth'] +
            achievement_complexity['overall_score'] * weights['achievement_complexity']
        )
        
        # Apply years-based floor to prevent obvious misclassifications
        # Someone with 5+ years should NEVER be classified as junior
        # Someone with 8+ years should at minimum be mid-level
        # Principal level requires 10+ years minimum
        if total_years >= 10:
            min_level = 'senior'
        elif total_years >= 5:
            min_level = 'mid-level'
        else:
            min_level = 'junior'

        # Map composite score to seniority level
        # Principal requires both high score (>=0.7) AND minimum 10 years experience
        if composite_score >= 0.7:
            # Apply floor: principal requires 10+ years
            if total_years >= 10:
                return 'principal'
            elif min_level == 'senior':
                return 'senior'
            else:
                return min_level
        elif composite_score >= 0.5:
            return 'senior' if min_level == 'senior' else min_level
        elif composite_score >= 0.3:
            return 'mid-level' if min_level in ['junior', 'mid-level'] else min_level
        else:
            # Apply floor based on years
            if min_level == 'senior':
                return 'senior'
            elif min_level == 'mid-level':
                return 'mid-level'
            return 'junior'

    def _calculate_confidence(self, total_years: float, data_completeness: float,
                            signal_consistency: float) -> float:
        """Calculate confidence score for the seniority detection."""
        # Base confidence on data quality
        confidence = data_completeness * 0.4

        # Add consistency factor
        confidence += signal_consistency * 0.4

        # Add experience factor (more experience = higher confidence)
        confidence += min(total_years / 10.0, 0.2)

        return min(confidence, 1.0)

    def _assess_data_completeness(self, experiences: List[Dict], skills: Set[str]) -> float:
        """Assess the completeness of available data."""
        score = 0.0

        # Experience completeness (check both old and new field names)
        if experiences:
            exp_completeness = sum(
                1 for exp in experiences
                if (exp.get('title') or exp.get('title_line') or exp.get('role')) and 
                   (exp.get('description') or exp.get('body') or exp.get('snippet') or exp.get('raw'))
            ) / len(experiences)
            score += exp_completeness * 0.6

        # Skills completeness
        if skills:
            score += min(len(skills) / 20.0, 0.4)  # Assume 20 skills is comprehensive

        return score

    def _assess_signal_consistency(self, seniority_level: str, total_years: float,
                                 leadership_score: float, skill_depth_score: float) -> float:
        """Assess consistency between different seniority signals."""
        expected_ranges = self.SENIORITY_LEVELS[seniority_level]

        # Check if years match expected range
        years_match = (expected_ranges['min_years'] <= total_years <=
                      expected_ranges.get('max_years', float('inf')))

        # Check if leadership matches expectations
        expected_leadership = 0.7 if seniority_level in ['senior', 'principal'] else 0.3
        leadership_match = abs(leadership_score - expected_leadership) < 0.3

        # Check if skill depth matches expectations
        expected_skill_depth = 0.8 if seniority_level in ['senior', 'principal'] else 0.5
        skill_depth_match = abs(skill_depth_score - expected_skill_depth) < 0.3

        # Calculate overall consistency
        matches = sum([years_match, leadership_match, skill_depth_match])
        return matches / 3.0

    def _generate_reasoning(self, seniority_level: str, total_years: float,
                          experience_quality: Dict, leadership_score: float,
                          skill_depth_score: float, achievement_complexity: Dict) -> str:
        """Generate human-readable reasoning for the seniority determination."""
        reasons = []

        # Years of experience
        if total_years < 2:
            reasons.append(f"{total_years} years of experience indicates early career stage")
        elif total_years < 5:
            reasons.append(f"{total_years} years of experience suggests growing expertise")
        elif total_years < 10:
            reasons.append(f"{total_years} years of experience demonstrates significant expertise")
        else:
            reasons.append(f"{total_years} years of experience indicates extensive professional background")

        # Leadership experience
        if leadership_score > 0.7:
            reasons.append("Strong leadership experience with team management and strategic contributions")
        elif leadership_score > 0.3:
            reasons.append("Some leadership experience with project coordination and mentorship")

        # Technical depth
        if skill_depth_score > 0.7:
            reasons.append("Advanced technical skills with architecture and design capabilities")
        elif skill_depth_score > 0.4:
            reasons.append("Solid technical foundation with intermediate to advanced skills")

        # Achievement complexity
        if achievement_complexity['overall_score'] > 0.7:
            reasons.append("Complex achievements with significant scale and business impact")

        return " ".join(reasons)

    def _generate_recommendations(self, seniority_level: str, total_years: float,
                                skill_depth_score: float, leadership_score: float) -> List[str]:
        """Generate career development recommendations based on seniority analysis."""
        recommendations = []

        if seniority_level == 'junior':
            recommendations.extend([
                "Focus on building technical depth in core skills",
                "Seek opportunities for mentorship and code reviews",
                "Work on small to medium-sized features independently",
                "Develop understanding of system architecture"
            ])
        elif seniority_level == 'mid-level':
            recommendations.extend([
                "Take ownership of larger features or small projects",
                "Mentor junior team members",
                "Develop architectural decision-making skills",
                "Contribute to technical design discussions"
            ])
        elif seniority_level == 'senior':
            recommendations.extend([
                "Lead technical initiatives and architecture decisions",
                "Mentor and develop mid-level engineers",
                "Drive cross-team technical collaborations",
                "Contribute to engineering strategy and roadmap"
            ])
        elif seniority_level == 'principal':
            recommendations.extend([
                "Drive organization-wide technical strategy",
                "Mentor senior engineers and develop technical leaders",
                "Represent engineering in executive discussions",
                "Innovate in technical processes and methodologies"
            ])

        return recommendations

    # Helper methods
    def _detect_title_seniority(self, title: str) -> float:
        """Detect seniority indicators in job titles."""
        title_lower = title.lower()

        for level, config in self.SENIORITY_LEVELS.items():
            for typical_title in config['typical_titles']:
                if typical_title in title_lower:
                    # Return score based on seniority level
                    seniority_scores = {'junior': 0.2, 'mid-level': 0.5, 'senior': 0.8, 'principal': 1.0}
                    return seniority_scores.get(level, 0.5)

        return 0.5  # Default to mid-level if no indicators

    def _analyze_description_complexity(self, description: str) -> float:
        """Analyze the complexity level of experience descriptions."""
        # Count technical terms, action verbs, and complexity indicators
        technical_terms = len(re.findall(r'\b(api|framework|architecture|database|system|application)\b', description))
        action_verbs = len(re.findall(r'\b(developed|designed|implemented|architected|optimized)\b', description))
        complexity_indicators = len(re.findall(r'\b(scalable|distributed|complex|enterprise|large-scale)\b', description))

        total_indicators = technical_terms + action_verbs + complexity_indicators
        return min(total_indicators / 10.0, 1.0)  # Normalize to 0-1

    def _detect_achievement_metrics(self, description: str) -> float:
        """Detect quantifiable achievement metrics in descriptions."""
        # Look for numbers, percentages, and impact metrics
        metrics_found = []

        # Percentage improvements
        metrics_found.extend(re.findall(r'\d+%', description))

        # Number ranges (users, revenue, etc.)
        metrics_found.extend(re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', description))

        # Impact keywords with metrics
        impact_patterns = [
            r'reduced\s+\w+\s+by\s+\d+%',
            r'increased\s+\w+\s+by\s+\d+%',
            r'improved\s+\w+\s+by\s+\d+%',
            r'saved\s+\$?\d+',
            r'generated\s+\$?\d+',
        ]

        for pattern in impact_patterns:
            metrics_found.extend(re.findall(pattern, description, re.IGNORECASE))

        return min(len(metrics_found) / 5.0, 1.0)  # Normalize to 0-1

    def _assess_technical_depth(self, description: str) -> float:
        """Assess the technical depth of experience descriptions."""
        depth_indicators = {
            'architecture': ['architecture', 'design', 'system design', 'solution'],
            'implementation': ['implemented', 'developed', 'coded', 'built'],
            'optimization': ['optimized', 'improved', 'enhanced', 'refactored'],
            'troubleshooting': ['debugged', 'resolved', 'fixed', 'troubleshot']
        }

        score = 0
        for category, indicators in depth_indicators.items():
            for indicator in indicators:
                if indicator in description.lower():
                    score += 0.25
                    break

        return min(score, 1.0)

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text."""
        numbers = []
        for match in re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text):
            try:
                numbers.append(float(match.replace(',', '')))
            except ValueError:
                continue
        return numbers


class ExperienceParser:
    """Helper class for parsing experience durations and dates."""

    def extract_duration_from_text(self, text: str) -> str:
        """Extract duration information from unstructured text."""
        # Look for common date patterns
        date_patterns = [
            r'\d{1,2}/\d{4}\s*-\s*(?:\d{1,2}/\d{4}|present|current)',
            r'\w+\s+\d{4}\s*-\s*(?:\w+\s+\d{4}|present|current)',
            r'\d{4}\s*-\s*(?:\d{4}|present|current)'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()

        return ""

    def parse_duration_to_months(self, duration: str) -> int:
        """Convert duration string to months."""
        if not duration:
            return 0

        # Try to extract years and months
        years_match = re.search(r'(\d+)\s*years?', duration, re.IGNORECASE)
        months_match = re.search(r'(\d+)\s*months?', duration, re.IGNORECASE)

        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0

        # If no explicit duration, try to parse date ranges
        if years == 0 and months == 0:
            date_range = self.extract_duration_from_text(duration)
            if date_range:
                years = self._calculate_years_from_date_range(date_range)

        return years * 12 + months

    def _calculate_years_from_date_range(self, date_range: str) -> int:
        """Calculate years from a date range string."""
        from datetime import datetime
        
        try:
            date_range_lower = date_range.lower()
            
            # Extract all years from the string
            dates = re.findall(r'\d{4}', date_range)
            
            if len(dates) >= 2:
                # Two explicit years: "2016 - 2019"
                start_year = int(dates[0])
                end_year = int(dates[1])
                return max(0, end_year - start_year)
            elif len(dates) == 1:
                start_year = int(dates[0])
                # Check if it ends with "present" or "current"
                if 'present' in date_range_lower or 'current' in date_range_lower:
                    current_year = datetime.now().year
                    return max(0, current_year - start_year)
                else:
                    # Single date with no "present", assume it's a 1-year role
                    return 1
        except (ValueError, IndexError):
            pass

        return 0


# Example usage and testing
if __name__ == "__main__":
    detector = SeniorityDetector()

    # Test with sample data
    sample_experiences = [
        {
            'title': 'Senior Software Engineer',
            'duration': '2019-2024',
            'description': 'Led team of 5 engineers. Architected scalable microservices processing 1M+ requests/day. Mentored junior developers.'
        },
        {
            'title': 'Software Engineer',
            'duration': '2016-2019',
            'description': 'Developed and maintained web applications. Improved performance by 40%.'
        }
    ]

    sample_skills = {'python', 'aws', 'microservices', 'team leadership', 'system design'}

    result = detector.detect_seniority(sample_experiences, sample_skills)
    print(json.dumps(result, indent=2))