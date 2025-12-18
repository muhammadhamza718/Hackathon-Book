/**
 * Signin Page
 *
 * Public page for user authentication.
 * Uses Docusaurus Layout for consistent theme integration.
 */

import React from "react";
import Layout from "@theme/Layout";
import SigninForm from "@site/src/components/auth/SigninForm";

export default function SigninPage() {
  return (
    <Layout
      title="Sign In"
      description="Sign in to access your personalized Physical AI & Humanoid Robotics learning experience"
    >
      <main style={{ minHeight: "calc(100vh - 60px)" }}>
        <SigninForm />
      </main>
    </Layout>
  );
}
