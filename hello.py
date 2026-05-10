def main():
    print("Hello from tinygpt!")


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    print(os.getenv("url"))
    main()
