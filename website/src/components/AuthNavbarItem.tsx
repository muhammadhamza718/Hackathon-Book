import React, { useEffect, useState } from "react";
import Link from "@docusaurus/Link";
import { useLocation } from "@docusaurus/router"; // Import useLocation
import NavbarUserMenu from "@/components/NavbarUserMenu";
import useAuth from "@/hooks/useAuth";
import styles from "./AuthNavbarItem.module.css";

// Prevent hydration mismatch by rendering only after mount
const BrowserOnly = ({ children }: { children: React.ReactNode }) => {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div style={{ width: 100 }} />; // Placeholder to reduce layout shift
  }
  return <>{children}</>;
};

export default function AuthNavbarItem() {
  const { user, signout, loading } = useAuth();
  const location = useLocation(); // Get current location

  // Don't render auth buttons on signup/login pages to avoid clutter
  if (location.pathname === "/signup" || location.pathname === "/signin") {
    return null;
  }

  const handleLogout = async () => {
    await signout();
  };

  return (
    <BrowserOnly>
      {loading ? (
        <div className={styles.loadingPlaceholder} />
      ) : user ? (
        <NavbarUserMenu user={user} onLogout={handleLogout} />
      ) : (
        <div className={styles.authButtons}>
          <Link to="/signin" className={styles.signinBtn}>
            Sign In
          </Link>
          <Link to="/signup" className={styles.signupBtn}>
            Sign Up
            <span className={styles.shine} />
          </Link>
        </div>
      )}
    </BrowserOnly>
  );
}
