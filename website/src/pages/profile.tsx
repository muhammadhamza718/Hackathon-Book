/**
 * Profile Page
 *
 * Displays user information with improved aesthetics and alignment.
 */

import React from "react";
import Layout from "@theme/Layout";
import useAuth from "@/hooks/useAuth";
import { Redirect } from "@docusaurus/router";

export default function ProfilePage() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <Layout>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            minHeight: "50vh",
            color: "var(--ifm-color-primary)",
            fontSize: "1.2rem",
            fontFamily: "Audiowide, sans-serif",
          }}
        >
          <div
            className="spinner-loading"
            style={{ marginRight: "1rem" }}
          ></div>
          Accessing Neural Profile...
        </div>
      </Layout>
    );
  }

  if (!user) {
    return <Redirect to="/signin" />;
  }

  return (
    <Layout title="User Profile">
      <main className="container margin-vert--xl" style={{ maxWidth: "800px" }}>
        <h1
          className="margin-bottom--lg"
          style={{
            fontFamily: "Audiowide, sans-serif",
            fontSize: "2.5rem",
            textShadow: "0 0 10px rgba(0, 243, 255, 0.3)",
          }}
        >
          User Profile
        </h1>

        {/* Basic Information Section */}
        <div className="card margin-bottom--lg shadow--md">
          <div
            className="card__header"
            style={{
              background: "rgba(0, 243, 255, 0.05)",
              borderBottom: "1px solid rgba(0, 243, 255, 0.1)",
              padding: "1rem 1.5rem",
            }}
          >
            <h3 style={{ margin: 0, color: "var(--ifm-color-primary)" }}>
              Basic Information
            </h3>
          </div>
          <div className="card__body" style={{ padding: "1.5rem" }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "150px 1fr",
                gap: "1rem",
              }}
            >
              <div style={{ fontWeight: 700, opacity: 0.7 }}>Name:</div>
              <div style={{ fontSize: "1.1rem" }}>{user.name}</div>

              <div style={{ fontWeight: 700, opacity: 0.7 }}>Email:</div>
              <div style={{ fontSize: "1.1rem" }}>{user.email}</div>

              <div style={{ fontWeight: 700, opacity: 0.7 }}>Account ID:</div>
              <div
                style={{
                  fontFamily: "monospace",
                  background: "rgba(0,0,0,0.3)",
                  padding: "2px 8px",
                  borderRadius: "4px",
                  fontSize: "0.85rem",
                  wordBreak: "break-all",
                }}
              >
                {user.id}
              </div>
            </div>
          </div>
        </div>

        {/* Neural Network / Background Section */}
        <div className="card shadow--md">
          <div
            className="card__header"
            style={{
              background: "rgba(188, 19, 254, 0.05)",
              borderBottom: "1px solid rgba(188, 19, 254, 0.1)",
              padding: "1rem 1.5rem",
            }}
          >
            <h3 style={{ margin: 0, color: "#bc13fe" }}>Learning Background</h3>
          </div>
          <div className="card__body" style={{ padding: "1.5rem" }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "200px 1fr",
                gap: "1.5rem",
              }}
            >
              <div style={{ fontWeight: 700, opacity: 0.7 }}>
                Software Experience:
              </div>
              <div style={{ textTransform: "capitalize", fontWeight: 500 }}>
                {user.softwareExperience || "Not specified"}
              </div>

              <div style={{ fontWeight: 700, opacity: 0.7 }}>
                AI/ML Familiarity:
              </div>
              <div style={{ textTransform: "capitalize", fontWeight: 500 }}>
                {user.aiMlFamiliarity || "Not specified"}
              </div>

              <div style={{ fontWeight: 700, opacity: 0.7 }}>
                Hardware Experience:
              </div>
              <div style={{ textTransform: "capitalize", fontWeight: 500 }}>
                {user.hardwareExperience || "Not specified"}
              </div>

              <div style={{ fontWeight: 700, opacity: 0.7 }}>
                Learning Goals:
              </div>
              <div style={{ textTransform: "capitalize", fontWeight: 500 }}>
                {user.learningGoals?.replace("-", " ") || "Not specified"}
              </div>

              {user.programmingLanguages && (
                <>
                  <div style={{ fontWeight: 700, opacity: 0.7 }}>
                    Programming Languages:
                  </div>
                  <div
                    style={{
                      background: "rgba(0, 243, 255, 0.1)",
                      padding: "0.5rem 1rem",
                      borderRadius: "0.5rem",
                      border: "1px solid rgba(0, 243, 255, 0.2)",
                    }}
                  >
                    {user.programmingLanguages}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </main>

      <style jsx>{`
        .spinner-loading {
          width: 20px;
          height: 20px;
          border: 2px solid rgba(0, 243, 255, 0.3);
          border-top: 2px solid var(--ifm-color-primary);
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          0% {
            transform: rotate(0deg);
          }
          100% {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </Layout>
  );
}
