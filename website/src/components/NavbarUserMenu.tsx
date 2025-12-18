/**
 * NavbarUserMenu Component
 *
 * Displays user information and account actions in the navbar.
 * Shows when user is authenticated.
 */

import React, { useState, useRef, useEffect } from "react";
import { useHistory } from "@docusaurus/router";
import { authClient } from "@/lib/auth-client";
type User = typeof authClient.$Infer.Session.user;

import styles from "./NavbarUserMenu.module.css";

interface NavbarUserMenuProps {
  user: User;
  onLogout: () => Promise<void>;
}

export default function NavbarUserMenu({
  user,
  onLogout,
}: NavbarUserMenuProps) {
  const history = useHistory();
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLogout = async () => {
    setLoading(true);
    try {
      await onLogout();
      setIsOpen(false);
      history.push("/");
    } catch (err) {
      console.error("Logout error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.userMenuContainer} ref={menuRef}>
      <button
        className={styles.userButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="User menu"
        aria-expanded={isOpen}
      >
        <div className={styles.avatar}>
          {user.name?.charAt(0).toUpperCase() ||
            user.email?.charAt(0).toUpperCase() ||
            "U"}
        </div>
        <span className={styles.userName}>{user.name || user.email}</span>
        <svg
          className={`${styles.chevron} ${isOpen ? styles.chevronOpen : ""}`}
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M2 4L6 8L10 4"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {isOpen && (
        <div className={styles.dropdown}>
          <div className={styles.dropdownHeader}>
            <div className={styles.avatarLarge}>
              {user.name?.charAt(0).toUpperCase() ||
                user.email?.charAt(0).toUpperCase() ||
                "U"}
            </div>
            <div className={styles.userInfo}>
              <div className={styles.userNameLarge}>{user.name || "User"}</div>
              <div className={styles.userEmail}>{user.email}</div>
            </div>
          </div>

          <div className={styles.separator}></div>

          <button
            className={styles.menuItem}
            onClick={() => {
              history.push("/profile");
              setIsOpen(false);
            }}
          >
            <span className={styles.menuIcon}>👤</span>
            Profile Settings
          </button>

          <div className={styles.separator}></div>

          <button
            className={`${styles.menuItem} ${styles.logoutButton}`}
            onClick={handleLogout}
            disabled={loading}
          >
            <span className={styles.menuIcon}>🚪</span>
            {loading ? "Signing Out..." : "Sign Out"}
          </button>
        </div>
      )}
    </div>
  );
}
