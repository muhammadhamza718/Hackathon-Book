/**
 * Signup Page
 *
 * Public page for new user registration.
 * Uses Docusaurus Layout for consistent theme integration.
 */

import React from "react";
import Layout from "@theme/Layout";
import SignupForm from "@site/src/components/auth/SignupForm";

export default function SignupPage() {
  return (
    <Layout
      title="Sign Up"
      description="Create your account to access personalized Physical AI & Humanoid Robotics content"
    >
      <main
        style={{
          minHeight: "calc(100vh - 60px)",
          paddingTop: "2rem",
          paddingBottom: "2rem",
        }}
      >
        <SignupForm />
      </main>
    </Layout>
  );
}
