import React from 'react'
import { DocsThemeConfig } from 'nextra-theme-docs'
import { ApeLogo } from './components/ApeLogo'

const footerNav = [{ name: 'Contact', href: 'mailto:founders@weavel.ai' }]

const config: DocsThemeConfig = {
  logo: (
    <div className="flex flex-row items-center gap-2">
      <ApeLogo width={48} height={48} />
      <span className="text-xl font-bold font-mono">Ape</span>
    </div>
  ),
  project: {
    link: 'https://github.com/weavel-ai/Ape'
  },
  chat: {
    link: 'https://weavel.ai/discord'
  },
  docsRepositoryBase: 'https://github.com/weavel-ai/Ape',
  editLink: {
    content: 'Edit this page on GitHub →',
    component: ({ children, ...props }) => (
      <a
        target="_blank"
        rel="noopener noreferrer"
        href={`https://github.com/weavel-ai/Ape/blob/main/docs/${props.filePath}`}
        {...props}
      >
        {children}
      </a>
    )
  },
  footer: {
    content: props => (
      <div className="flex flex-col lg:flex-row items-center gap-5 text-sm w-full">
        <span className="text-base-content text-sm">
          © {new Date().getFullYear()} Weavel, Inc. All rights reserved.
        </span>
        <div className="grow" />
        {footerNav.map(nav => (
          <a
            key={nav.name}
            href={nav.href}
            target="_blank"
            rel="noopener noreferrer"
            className={
              'inline rounded-none leading-6 transition-all text-base-content hover:text-primary whitespace-nowrap hover:underline'
            }
          >
            {nav.name}
          </a>
        ))}
        <a
          target="_blank"
          rel="noopener noreferrer"
          href="https://x.com/weaveldotai"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            fill="currentColor"
            viewBox="0 0 16 16"
          >
            <path d="M12.6.75h2.454l-5.36 6.142L16 15.25h-4.937l-3.867-5.07-4.425 5.07H.316l5.733-6.57L0 .75h5.063l3.495 4.633L12.601.75Zm-.86 13.028h1.36L4.323 2.145H2.865z" />
          </svg>
        </a>
        <a
          target="_blank"
          rel="noopener noreferrer"
          href="https://www.linkedin.com/company/weavel"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="lucide lucide-linkedin size-5"
          >
            <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
            <rect width="4" height="12" x="2" y="9" />
            <circle cx="4" cy="4" r="2" />
          </svg>
        </a>
      </div>
    )
  }
}

export default config
