/**
 * SignupForm Component
 *
 * User registration form with email/password and background questionnaire.
 * Features: validation, loading states, error handling, glassmorphism styling.
 */

import React, { useState } from "react";
import { useHistory } from "@docusaurus/router";
import useAuth from "@/hooks/useAuth";
import BackgroundQuestionnaire from "./BackgroundQuestionnaire";
import styles from "./SignupForm.module.css";

export default function SignupForm() {
  const history = useHistory();
  const { signup, loading, error } = useAuth();

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    softwareExperience: "",
    aiMlFamiliarity: "",
    hardwareExperience: "",
    learningGoals: "",
    programmingLanguages: "",
  });

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [validationErrors, setValidationErrors] = useState<
    Record<string, string>
  >({});
  const [debugError, setDebugError] = useState<any>(null);
  const [formError, setFormError] = useState("");

  const handleChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    // Clear validation error for this field when user starts typing
    if (validationErrors[field]) {
      setValidationErrors((prev) => {
        const newErrors = { ...prev };
        delete newErrors[field];
        return newErrors;
      });
    }
  };

  const validateForm = (): boolean => {
    const errors: Record<string, string> = {};

    if (!formData.name.trim()) {
      errors.name = "Name is required";
    }

    if (!formData.email.trim()) {
      errors.email = "Email is required";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      errors.email = "Please enter a valid email address";
    }

    if (!formData.password) {
      errors.password = "Password is required";
    } else if (formData.password.length < 8) {
      errors.password = "Password must be at least 8 characters";
    }

    if (formData.password !== formData.confirmPassword) {
      errors.confirmPassword = "Passwords do not match";
    }

    if (!formData.softwareExperience) {
      errors.softwareExperience = "Please select your experience level";
    }

    if (!formData.aiMlFamiliarity) {
      errors.aiMlFamiliarity = "Please select your AI/ML familiarity";
    }

    if (!formData.hardwareExperience) {
      errors.hardwareExperience = "Please select your hardware experience";
    }

    if (!formData.learningGoals) {
      errors.learningGoals = "Please select your learning goal";
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError("");
    setDebugError(null);

    if (!validateForm()) {
      return;
    }

    try {
      await signup(formData.email, formData.password, formData.name, {
        softwareExperience: formData.softwareExperience,
        aiMlFamiliarity: formData.aiMlFamiliarity,
        hardwareExperience: formData.hardwareExperience,
        learningGoals: formData.learningGoals,
        programmingLanguages: formData.programmingLanguages,
      });

      // Redirect to homepage on successful signup
      history.push("/");
    } catch (err) {
      const errorObj = {
        message: err instanceof Error ? err.message : String(err),
        stack: err instanceof Error ? err.stack : undefined,
        fullError: err,
      };

      console.error("Signup error details:", errorObj);
      setDebugError(errorObj);

      const errorMessage =
        err instanceof Error ? err.message : "Signup failed. Please try again.";
      // Provide more specific error messages for common issues
      let userFacingMessage = errorMessage;
      if (
        errorMessage.includes("duplicate") ||
        errorMessage.includes("already exists")
      ) {
        userFacingMessage =
          "An account with this email already exists. Please try signing in instead.";
      } else if (
        errorMessage.includes("network") ||
        errorMessage.includes("fetch")
      ) {
        userFacingMessage =
          "Unable to connect to the server. Please check your internet connection and try again.";
      } else if (errorMessage.includes("timeout")) {
        userFacingMessage = "The request timed out. Please try again.";
      }
      setFormError(userFacingMessage);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.signupForm}>
      <div className={styles.formHeader}>
        <h2 className={styles.title}>Create Account</h2>
        <p className={styles.subtitle}>
          Join the Physical AI & Humanoid Robotics community
        </p>
      </div>

      {(formError || error) && (
        <div className={styles.errorBanner}>
          <div className={styles.errorSummary}>⚠️ {formError || error}</div>
          {/* Debug Details - remove in production */}
          <details
            className={styles.debugDetails}
            style={{
              marginTop: "10px",
              fontSize: "12px",
              color: "#ff6b6b",
              textAlign: "left",
            }}
          >
            <summary>Show Error Details</summary>
            <pre
              style={{
                whiteSpace: "pre-wrap",
                overflowX: "auto",
                background: "rgba(0,0,0,0.2)",
                padding: "10px",
                borderRadius: "5px",
              }}
            >
              {JSON.stringify(debugError, null, 2)}
            </pre>
          </details>
        </div>
      )}

      {/* Basic Information */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Basic Information</h3>

        <div className={styles.formGroup}>
          <label htmlFor="name" className={styles.label}>
            Full Name
          </label>
          <input
            type="text"
            id="name"
            value={formData.name}
            onChange={(e) => handleChange("name", e.target.value)}
            className={`${styles.input} ${
              validationErrors.name ? styles.error : ""
            }`}
            placeholder="Enter your full name"
            disabled={loading}
          />
          {validationErrors.name && (
            <span className={styles.errorMessage}>{validationErrors.name}</span>
          )}
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="email" className={styles.label}>
            Email Address
          </label>
          <input
            type="email"
            id="email"
            value={formData.email}
            onChange={(e) => handleChange("email", e.target.value)}
            className={`${styles.input} ${
              validationErrors.email ? styles.error : ""
            }`}
            placeholder="your.email@example.com"
            disabled={loading}
          />
          {validationErrors.email && (
            <span className={styles.errorMessage}>
              {validationErrors.email}
            </span>
          )}
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="password" className={styles.label}>
            Password
          </label>
          <div className={styles.passwordWrapper}>
            <input
              type={showPassword ? "text" : "password"}
              id="password"
              value={formData.password}
              onChange={(e) => handleChange("password", e.target.value)}
              className={`${styles.input} ${
                validationErrors.password ? styles.error : ""
              }`}
              placeholder="At least 8 characters"
              disabled={loading}
            />
            <button
              type="button"
              className={styles.togglePassword}
              onClick={() => setShowPassword(!showPassword)}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? "👁️" : "👁️‍🗨️"}
            </button>
          </div>
          {validationErrors.password && (
            <span className={styles.errorMessage}>
              {validationErrors.password}
            </span>
          )}
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="confirmPassword" className={styles.label}>
            Confirm Password
          </label>
          <div className={styles.passwordWrapper}>
            <input
              type={showConfirmPassword ? "text" : "password"}
              id="confirmPassword"
              value={formData.confirmPassword}
              onChange={(e) => handleChange("confirmPassword", e.target.value)}
              className={`${styles.input} ${
                validationErrors.confirmPassword ? styles.error : ""
              }`}
              placeholder="Re-enter your password"
              disabled={loading}
            />
            <button
              type="button"
              className={styles.togglePassword}
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              aria-label={
                showConfirmPassword ? "Hide password" : "Show password"
              }
            >
              {showConfirmPassword ? "👁️" : "👁️‍🗨️"}
            </button>
          </div>
          {validationErrors.confirmPassword && (
            <span className={styles.errorMessage}>
              {validationErrors.confirmPassword}
            </span>
          )}
        </div>
      </div>

      {/* Background Questionnaire */}
      <BackgroundQuestionnaire
        values={{
          softwareExperience: formData.softwareExperience,
          aiMlFamiliarity: formData.aiMlFamiliarity,
          hardwareExperience: formData.hardwareExperience,
          learningGoals: formData.learningGoals,
          programmingLanguages: formData.programmingLanguages,
        }}
        onChange={handleChange}
        errors={validationErrors}
      />

      {/* Submit Button */}
      <button type="submit" className={styles.submitButton} disabled={loading}>
        {loading ? (
          <>
            <span className={styles.spinner}></span>
            Creating Account...
          </>
        ) : (
          "Create Account"
        )}
      </button>

      <div className={styles.footer}>
        Already have an account?{" "}
        <a href="/signin" className={styles.link}>
          Sign In
        </a>
      </div>
    </form>
  );
}
