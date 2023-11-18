import nltk
from nltk import FreqDist
import matplotlib.pyplot as plt

nltk.download('punkt')

def calculate_burstiness(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Calculate word frequency distribution
    word_frequency = FreqDist(words)

    # Calculate burstiness using Gini coefficient
    n = len(words)
    gini_coefficient = 1 - sum([(word_frequency[word] / n) ** 2 for word in word_frequency]) / (n ** 2)
    burstiness = 2 * gini_coefficient

    return burstiness

def plot_word_frequency_distribution(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Calculate word frequency distribution
    word_frequency = FreqDist(words)

    # Plot word frequency distribution
    word_frequency.plot(30, cumulative=False)
    plt.show()

def main():
    print("Welcome to the Burstiness Calculator!")
    user_input = input("Enter the text you want to analyze: ")

    # Calculate and display burstiness
    burstiness = calculate_burstiness(user_input)
    print(f"Burstiness: {burstiness:.2%}")

    # Plot word frequency distribution
    plot_word_frequency_distribution(user_input)

if __name__ == "__main__":
    main()
