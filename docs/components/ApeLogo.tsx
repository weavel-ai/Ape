import Image from 'next/image'

export function ApeLogo({
  className,
  ...props
}: Omit<React.ComponentProps<typeof Image>, 'src' | 'alt'>) {
  return (
    <>
      <Image
        draggable={false}
        src="/ape-light.svg"
        alt="Ape Logo"
        width={300}
        height={300}
        className={`dark:hidden ${className}`}
        {...props}
      />
      <Image
        draggable={false}
        src="/ape-dark.svg"
        alt="Ape Logo"
        width={300}
        height={300}
        className={`hidden dark:block ${className}`}
        {...props}
      />
    </>
  )
}
