import os


def generate_allcode(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('PROJECT STRUCTURE:\n')
        f.write('==================\n')
        for root, dirs, files in os.walk(root_dir):
            if '__pycache__' in root or '.git' in root:
                continue
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write('{}{}/\n'.format(indent, os.path.basename(root)))
            sub_indent = ' ' * 4 * (level + 1)
            for file in sorted(files):
                if file == 'allcode.txt':
                    continue
                if file.endswith('.pyc'):
                    continue
                f.write('{}{}\n'.format(sub_indent, file))
        f.write('\n\nFILE CONTENTS:\n')
        f.write('==============\n')
        for root, dirs, files in os.walk(root_dir):
            if '__pycache__' in root or '.git' in root:
                continue
            for file in sorted(files):
                if file == 'allcode.txt':
                    continue
                if file.endswith('.pyc'):
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as file_content:
                        content = file_content.read()
                    f.write('\n' + '='*80 + '\n')
                    f.write(f'FILE: {os.path.relpath(file_path, root_dir)}\n')
                    f.write('='*80 + '\n')
                    f.write(content)
                    f.write('\n')
                except Exception as e:
                    f.write(f'\nCould not read {file_path}: {e}\n')


if __name__ == '__main__':
    generate_allcode(os.getcwd(), 'allcode.txt')
