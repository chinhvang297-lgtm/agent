import re
import os

files_to_fix = [
    '/home/s26-shijiarong/PPTAgent/pptagent/presentation/layout.py',
    '/home/s26-shijiarong/PPTAgent/pptagent/pptgen.py', 
    '/home/s26-shijiarong/PPTAgent/pptagent/induct.py',
    '/home/s26-shijiarong/PPTAgent/pptagent/multimodal.py',
    '/home/s26-shijiarong/PPTAgent/pptagent/document/document.py'
]

for file_path in files_to_fix:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace TaskGroup with gather
        content = re.sub(
            r'async with asyncio\.TaskGroup\(\) as tg:\s*\n(.*?)tg\.create_task\((.*?)\)',
            r'tasks = []\n\1tasks.append(\2)',
            content,
            flags=re.DOTALL
        )
        
        # Add await asyncio.gather after task collection
        content = re.sub(
            r'(tasks\.append\(.*?\))\s*\n',
            r'\1\n        await asyncio.gather(*tasks)\n',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f'Fixed {file_path}')

print('TaskGroup compatibility fix completed')
