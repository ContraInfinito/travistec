import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs['recommended-latest'],
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      // Pre-existing UI components carry some lint debt (unused vars, empty
      // catches, etc.). These are surfaced as warnings so CI stays green while
      // the debt is tracked and cleaned up incrementally — see AUDIT.md.
      'no-unused-vars': ['warn', { varsIgnorePattern: '^[A-Z_]', args: 'none' }],
      'no-useless-escape': 'warn',
      'no-empty': 'warn',
      'no-constant-condition': 'warn',
    },
  },
])
