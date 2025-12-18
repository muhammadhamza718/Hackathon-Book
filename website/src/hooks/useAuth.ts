/**
 * useAuth Hook
 *
 * Custom React hook for authentication state management and operations.
 * Wraps standard Better Auth hooks and ensures type safety.
 */

import { useCallback } from "react";
import { authClient, signIn, signUp, useSession } from "@/lib/auth-client";

// Define the shape of the background profile for typescript safety
interface BackgroundProfile {
  softwareExperience: string;
  aiMlFamiliarity: string;
  hardwareExperience: string;
  learningGoals: string;
  programmingLanguages?: string;
}

// Inferred types from the client
type User = typeof authClient.$Infer.Session.user;
type Session = typeof authClient.$Infer.Session.session;

interface UseAuthReturn {
  user: User | null;
  session: Session | null;
  loading: boolean;
  error: string | null;
  signup: (
    email: string,
    password: string,
    name: string,
    profile: BackgroundProfile
  ) => Promise<void>;
  signin: (email: string, password: string) => Promise<void>;
  signout: () => Promise<void>;
  updateProfile: (profile: Partial<BackgroundProfile>) => Promise<void>;
  checkSession: () => Promise<void>;
}

export default function useAuth(): UseAuthReturn {
  const { data, isPending, error: sessionError, refetch } = useSession();

  const loading = isPending;
  const error = sessionError ? sessionError.message : null;
  const user = data?.user || null;
  const session = data?.session || null;

  const signup = useCallback(
    async (
      email: string,
      password: string,
      name: string,
      profile: BackgroundProfile
    ) => {
      const result = await signUp.email({
        email,
        password,
        name,
        ...profile,
      });

      if (result.error) {
        console.error("Better Auth signup error:", result.error);
        // Create a more descriptive error message
        const detailedMessage =
          result.error.message ||
          (result.error as any)?.statusText ||
          JSON.stringify(result.error);
        throw new Error(detailedMessage || "Signup failed");
      }
    },
    []
  );

  const signin = useCallback(async (email: string, password: string) => {
    const result = await signIn.email({
      email,
      password,
    });

    if (result.error) {
      console.error("Better Auth signin error:", result.error); // Log for debugging
      throw new Error(result.error.message || "Signin failed");
    }
  }, []);

  const signout = useCallback(async () => {
    const result = await authClient.signOut();
    if (result.error) {
      console.error("Better Auth signout error:", result.error); // Log for debugging
      throw new Error(result.error.message || "Signout failed");
    }
  }, []);

  const updateProfile = useCallback(
    async (profile: Partial<BackgroundProfile>) => {
      // Better Auth updateUser handles additional fields if configured in schema
      const result = await authClient.updateUser({
        ...profile,
      });

      if (result.error) {
        console.error("Better Auth update profile error:", result.error); // Log for debugging
        throw new Error(result.error.message || "Update failed");
      }

      // Refresh session to reflect changes
      refetch();
    },
    [refetch]
  );

  return {
    user,
    session,
    loading,
    error,
    signup,
    signin,
    signout,
    updateProfile,
    checkSession: async () => {
      refetch();
    }, // bridge for legacy calls
  };
}
