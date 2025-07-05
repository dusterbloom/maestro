// ui/design-system.ts
// Centralized design tokens for UI consistency

export const DESIGN_TOKENS = {
  // Colors
  primary: {
    default: '#3b82f6',
    hover: '#2563eb',
    active: '#1d4ed8'
  },
  secondary: {
    default: '#6b7280',
    hover: '#52525b',
    active: '#43434a'
  },
  background: {
    default: 'bg-white/70',
    card: 'bg-white/50',
    accent: 'from-blue-50 to-indigo-100'
  },
  
  // Spacing
  spacing: {
    small: 'space-y-2',
    medium: 'space-y-4',
    large: 'space-y-6'
  },
  
  // Typography
  heading: 'text-4xl font-bold text-gray-800',
  body: 'text-gray-600',
  accent: 'font-medium text-blue-800'
} as const;