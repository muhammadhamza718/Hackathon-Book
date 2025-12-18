/**
 * SigninForm Component
 *
 * User authentication form with email/password.
 * Features: validation, loading states, error handling, glassmorphism styling.
 */

import React, { useState, useEffect } from "react";
import { useHistory } from "@docusaurus/router";
import useAuth from "@/hooks/useAuth";
import styles from "./SigninForm.module.css";

export default function SigninForm() {
  const history = useHistory();
  const { signin, user, loading, error } = useAuth();

  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const [showPassword, setShowPassword] = useState(false);
  const [formError, setFormError] = useState("");

  // Redirect if already logged in
  useEffect(() => {
    if (user) {
      history.push("/");
    }
  }, [user, history]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setFormError(""); // Clear errors when user types
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError("");

    if (!formData.email || !formData.password) {
      setFormError("Please enter both email and password");
      return;
    }

    try {
      await signin(formData.email, formData.password);
      // Redirect handled by useEffect when user state updates
    } catch (err) {
      setFormError("Invalid email or password. Please try again.");
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.signinForm}>
      <div className={styles.formHeader}>
        <h2 className={styles.title}>Welcome Back</h2>
        <p className={styles.subtitle}>
          Sign in to continue your learning journey
        </p>
      </div>

      {(formError || error) && (
        <div className={styles.errorBanner}>⚠️ {formError || error}</div>
      )}

      <div className={styles.formGroup}>
        <label htmlFor="email" className={styles.label}>
          Email Address
        </label>
        <input
          type="email"
          id="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          className={styles.input}
          placeholder="your.email@example.com"
          disabled={loading}
          autoComplete="email"
          required
        />
      </div>

      <div className={styles.formGroup}>
        <label htmlFor="password" className={styles.label}>
          Password
        </label>
        <div className={styles.passwordWrapper}>
          <input
            type={showPassword ? "text" : "password"}
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            className={styles.input}
            placeholder="Enter your password"
            disabled={loading}
            autoComplete="current-password"
            required
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
      </div>

      <button type="submit" className={styles.submitButton} disabled={loading}>
        {loading ? (
          <>
            <span className={styles.spinner}></span>
            Signing In...
          </>
        ) : (
          "Sign In"
        )}
      </button>

      <div className={styles.footer}>
        Don't have an account?{" "}
        <a href="/signup" className={styles.link}>
          Create one
        </a>
      </div>
    </form>
  );
}
