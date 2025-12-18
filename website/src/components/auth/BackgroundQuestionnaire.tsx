/**
 * Background Questionnaire Component
 *
 * Collects user's technical background information during signup.
 * Styled with glassmorphism effect to match the cyberpunk theme.
 */

import React from "react";
import styles from "./BackgroundQuestionnaire.module.css";

interface BackgroundQuestionnaireProps {
  values: {
    softwareExperience: string;
    aiMlFamiliarity: string;
    hardwareExperience: string;
    learningGoals: string;
    programmingLanguages: string;
  };
  onChange: (field: string, value: string) => void;
  errors?: Record<string, string>;
}

export default function BackgroundQuestionnaire({
  values,
  onChange,
  errors = {},
}: BackgroundQuestionnaireProps) {
  return (
    <div className={styles.questionnaireContainer}>
      <h3 className={styles.questionnaireTitle}>About Your Background</h3>
      <p className={styles.questionnaireSubtitle}>
        Help us personalize your learning experience
      </p>

      {/* Software Experience */}
      <div className={styles.formGroup}>
        <label htmlFor="softwareExperience" className={styles.label}>
          Software Development Experience
          <span className={styles.tooltip}>
            How familiar are you with programming?
          </span>
        </label>
        <select
          id="softwareExperience"
          value={values.softwareExperience}
          onChange={(e) => onChange("softwareExperience", e.target.value)}
          className={`${styles.select} ${
            errors.softwareExperience ? styles.error : ""
          }`}
          required
        >
          <option value="">Select your level</option>
          <option value="beginner">Beginner - Just starting out</option>
          <option value="intermediate">Intermediate - Some experience</option>
          <option value="advanced">Advanced - Professional developer</option>
          <option value="expert">Expert - Senior/Lead developer</option>
        </select>
        {errors.softwareExperience && (
          <span className={styles.errorMessage}>
            {errors.softwareExperience}
          </span>
        )}
      </div>

      {/* AI/ML Familiarity */}
      <div className={styles.formGroup}>
        <label htmlFor="aiMlFamiliarity" className={styles.label}>
          AI/ML Knowledge
          <span className={styles.tooltip}>
            How well do you know artificial intelligence concepts?
          </span>
        </label>
        <select
          id="aiMlFamiliarity"
          value={values.aiMlFamiliarity}
          onChange={(e) => onChange("aiMlFamiliarity", e.target.value)}
          className={`${styles.select} ${
            errors.aiMlFamiliarity ? styles.error : ""
          }`}
          required
        >
          <option value="">Select your level</option>
          <option value="none">None - New to AI/ML</option>
          <option value="basic">Basic - Understand concepts</option>
          <option value="intermediate">Intermediate - Built some models</option>
          <option value="advanced">Advanced - Professional ML engineer</option>
        </select>
        {errors.aiMlFamiliarity && (
          <span className={styles.errorMessage}>{errors.aiMlFamiliarity}</span>
        )}
      </div>

      {/* Hardware/Robotics Experience */}
      <div className={styles.formGroup}>
        <label htmlFor="hardwareExperience" className={styles.label}>
          Hardware/Robotics Experience
          <span className={styles.tooltip}>
            Have you worked with physical robots or hardware?
          </span>
        </label>
        <select
          id="hardwareExperience"
          value={values.hardwareExperience}
          onChange={(e) => onChange("hardwareExperience", e.target.value)}
          className={`${styles.select} ${
            errors.hardwareExperience ? styles.error : ""
          }`}
          required
        >
          <option value="">Select your level</option>
          <option value="none">None - No hardware experience</option>
          <option value="hobbyist">Hobbyist - Personal projects</option>
          <option value="professional">
            Professional - Industry experience
          </option>
          <option value="educator">Educator - Teaching robotics</option>
        </select>
        {errors.hardwareExperience && (
          <span className={styles.errorMessage}>
            {errors.hardwareExperience}
          </span>
        )}
      </div>

      {/* Learning Goals */}
      <div className={styles.formGroup}>
        <label htmlFor="learningGoals" className={styles.label}>
          Primary Learning Goal
          <span className={styles.tooltip}>
            What's your main reason for learning?
          </span>
        </label>
        <select
          id="learningGoals"
          value={values.learningGoals}
          onChange={(e) => onChange("learningGoals", e.target.value)}
          className={`${styles.select} ${
            errors.learningGoals ? styles.error : ""
          }`}
          required
        >
          <option value="">Select your goal</option>
          <option value="career-change">
            Career Change - Switch to robotics
          </option>
          <option value="skill-upgrade">
            Skill Upgrade - Enhance existing skills
          </option>
          <option value="research">Research - Academic/scientific work</option>
          <option value="teaching">Teaching - Educate others</option>
          <option value="hobby">Hobby - Personal interest</option>
        </select>
        {errors.learningGoals && (
          <span className={styles.errorMessage}>{errors.learningGoals}</span>
        )}
      </div>

      {/* Programming Languages (Optional) */}
      <div className={styles.formGroup}>
        <label htmlFor="programmingLanguages" className={styles.label}>
          Familiar Programming Languages (Optional)
          <span className={styles.tooltip}>
            E.g., Python, C++, JavaScript (comma-separated)
          </span>
        </label>
        <input
          type="text"
          id="programmingLanguages"
          value={values.programmingLanguages}
          onChange={(e) => onChange("programmingLanguages", e.target.value)}
          className={styles.input}
          placeholder="Python, C++, JavaScript"
        />
      </div>
    </div>
  );
}
