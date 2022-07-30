from collections import defaultdict

class Trie:
    """
    Implement a trie with insert, search, and startsWith methods.
    """
    def __init__(self):
        self.root = defaultdict()

    # @param {string} word
    # @return {void}
    # Inserts a word into the trie.
    def insert(self, phrase):
        
        current = self.root
        for word in phrase.split():
            current = current.setdefault(word, {})
        current.setdefault(" [END] ")

    # @param {string} word
    # @return {boolean}
    # Returns if the word is in the trie.
    def search(self, phrase):
        current = self.root
        for word in phrase.split():
            if word not in current:
                return False
            current = current[word]
        if " [END] " in current:
            return True
        return False

    # @param {string} prefix
    # @return {boolean}
    # Returns if there is any word in the trie
    # that starts with the given prefix.
    def startsWith(self, phrase):
        current = self.root
        for word in phrase.split():
            if word not in current:
                return False
            current = current[word]
        if " [END] " in current:
            if len(current) == 1:
                return 'END ONLY'
            else:
                return 'END AND MORE'
        else:
            return 'NONE'

# Now test the class

test = Trie()
test.insert('hello world')
test.insert('i like apples')
test.insert('hello brother')
test.insert('hello world i like apples')

print(test.search('hello'))
print(test.search('i like apples'))
print(test.startsWith('hello'))
print(test.startsWith('hello world'))
print(test.startsWith('hello world i like apples'))