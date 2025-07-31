from tokenization.tokenizer import generate_token_map


if __name__ == "__main__":
    """
    Run this to export the token map
    """
    codebook_size: int = 64 
    generate_token_map(codebook_size)
