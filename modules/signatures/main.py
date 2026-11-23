from modules.signatures.signature_computer import *


def main():
    input_path = "../data/aligned_binary_discrepancies.csv"
    output_path = "../data/signatures_binary.csv"
    print()

    # Signature Computation
    computer = SignatureComputer(order=2)
    aligned = computer.extract(input_path)
    X, original_indices = computer.process(aligned)
    print(f"Computed signatures — shape {X.shape}")
    print(f"Signatures at \n position 10 : {X[10]}, \n postion 100 : {X[100]}, \n position 343 : {X[343]}")
    print()
    
    computer.export(X, original_indices, output_path)



if __name__ == "__main__":
    main()
