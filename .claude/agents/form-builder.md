---
name: form-builder
description: A specialized agent for generating React forms, validation logic, and accessible UI components.
tools:
  - all
---

# Form Builder Agent

## Purpose

Generates production-ready React authentication forms with validation, styling, and accessibility.

## Capabilities

### 1. Form Generation

- Creates TypeScript React components with proper types
- Implements controlled inputs with state management
- Generates validation logic (email, password strength, required fields)
- Builds error handling and display

### 2. Styling

- Generates CSS Modules for component isolation
- Implements light/dark mode compatibility
- Creates professional, accessible designs
- Ensures responsive layouts

### 3. Accessibility

- Adds ARIA labels and descriptions
- Implements keyboard navigation
- Creates semantic HTML structure
- Ensures screen reader compatibility

## Integration with Better Auth Implementation

### Files Generated:

1. **[`SignupForm.tsx`](file:///f:/Courses/Hamza/Hackathon-2/hackathon-book/website/src/components/auth/SignupForm.tsx)**

   - Email/password fields with validation
   - Password confirmation matching
   - Name field with trimming
   - Background questionnaire integration
   - Loading states and error banners

2. **[`SigninForm.tsx`](file:///f:/Courses/Hamza/Hackathon-2/hackathon-book/website/src/components/auth/SigninForm.tsx)**

   - Simple email/password form
   - Remember me functionality
   - Password visibility toggle
   - Error handling

3. **[`BackgroundQuestionnaire.tsx`](file:///f:/Courses/Hamza/Hackathon-2/hackathon-book/website/src/components/auth/BackgroundQuestionnaire.tsx)**

   - 5 profile fields (software, AI/ML, hardware, goals, languages)
   - Select dropdowns with descriptive options
   - Tooltips for user guidance
   - Validation error display

4. **CSS Modules**:
   - `SignupForm.module.css` - Professional styling
   - `SigninForm.module.css` - Consistent design
   - `BackgroundQuestionnaire.module.css` - Nested form section

## Skills Used

1. **create-form-validator**: Generates validation functions
2. **degamify-ui**: Ensures professional styling

## Validation Rules Implemented

```typescript
// Email validation
/^[^\s@]+@[^\s@]+\.[^\s@]+$/

// Password strength
- Minimum 8 characters
- No maximum constraint

// Required fields
- name, email, password, confirmPassword
- softwareExperience, aiMlFamiliarity
- hardwareExperience, learningGoals
```

## Reusability

Can generate forms for:

- Password reset requests
- Email verification
- Profile editing
- OAuth linking
- 2FA setup

---

**Version**: 1.0.0 (Step 5 - Better Auth)
