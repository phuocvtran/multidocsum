import underthesea


class Preprocessor:
    @staticmethods
    def get_content_from_raw(text: str) -> str:
        lines = text.split('\n')
        
        try:
            content_index = lines.index('Content:')
        except ValueError:
            for index, item in enumerate(lines):
                if item.count('Content:') != 0:
                    content_index = index - 1
                    lines[index] = lines[index].replace('Content:', '')
                    break
        return ' '.join(lines[content_index+1:-1])

    @staticmethods
    def sent_tokenize(text: str) -> str:
        return '\n'.join(underthesea.sent_tokenize(text))