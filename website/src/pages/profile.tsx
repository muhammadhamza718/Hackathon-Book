/**
 * Profile Page (MVP Stub)
 *
 * Displays user information.
 * Fully implemented in Phase 5.
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
          style={{ display: "flex", justifyContent: "center", padding: "4rem" }}
        >
          Loading profile...
        </div>
      </Layout>
    );
  }

  if (!user) {
    return <Redirect to="/signin" />;
  }

  return (
    <Layout title="User Profile">
      <main className="container margin-vert--lg">
        <h1>User Profile</h1>

        {/* Basic Information */}
        <div className="card margin-bottom--lg">
          <div className="card__header">
            <h3>Basic Information</h3>
          </div>
          <div className="card__body">
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <tbody>
                <tr>
                  <td style={{ padding: "0.5rem 0", fontWeight: 600 }}>
                    Name:
                  </td>
                  <td style={{ padding: "0.5rem 0" }}>{user.name}</td>
                </tr>
                <tr>
                  <td style={{ padding: "0.5rem 0", fontWeight: 600 }}>
                    Email:
                  </td>
                  <td style={{ padding: "0.5rem 0" }}>{user.email}</td>
                </tr>
                <tr>
                  <td style={{ padding: "0.5rem 0", fontWeight: 600 }}>
                    User ID:
                  </td>
                  <td
                    style={{
                      padding: "0.5rem 0",
                      fontFamily: "monospace",
                      fontSize: "0.9em",
                    }}
                  >
                    {user.id}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Background Questionnaire */}
        <div className="card">
          <div className="card__header">
            <h3>Learning Background</h3>
          </div>
          <div className="card__body">
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <tbody>
                <tr>
                  <td
                    style={{
                      padding: "0.5rem 0",
                      fontWeight: 600,
                      width: "40%",
                    }}
                  >
                    Software Experience:
                  </td>
                  <td
                    style={{ padding: "0.5rem 0", textTransform: "capitalize" }}
                  >
                    {user.softwareExperience || "Not specified"}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "0.5rem 0", fontWeight: 600 }}>
                    AI/ML Familiarity:
                  </td>
                  <td
                    style={{ padding: "0.5rem 0", textTransform: "capitalize" }}
                  >
                    {user.aiMlFamiliarity || "Not specified"}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "0.5rem 0", fontWeight: 600 }}>
                    Hardware Experience:
                  </td>
                  <td
                    style={{ padding: "0.5rem 0", textTransform: "capitalize" }}
                  >
                    {user.hardwareExperience || "Not specified"}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "0.5rem 0", fontWeight: 600 }}>
                    Learning Goals:
                  </td>
                  <td
                    style={{ padding: "0.5rem 0", textTransform: "capitalize" }}
                  >
                    {user.learningGoals?.replace("-", " ") || "Not specified"}
                  </td>
                </tr>
                {user.programmingLanguages && (
                  <tr>
                    <td style={{ padding: "0.5rem 0", fontWeight: 600 }}>
                      Programming Languages:
                    </td>
                    <td style={{ padding: "0.5rem 0" }}>
                      {user.programmingLanguages}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </Layout>
  );
}
