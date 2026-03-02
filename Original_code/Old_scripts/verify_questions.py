import json

with open('questions_with_chunks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Count questions by number of chunks
chunk_distribution = {}
for q in data['questions']:
    num_chunks = len(q['answer_chunk_ids'])
    chunk_distribution[num_chunks] = chunk_distribution.get(num_chunks, 0) + 1

print('Question Distribution by Chunk Count:')
for chunks in sorted(chunk_distribution.keys()):
    print(f'  {chunks} chunk(s): {chunk_distribution[chunks]} questions')
print(f'\nTotal questions: {sum(chunk_distribution.values())}')

# Print a few sample questions for each category
print('\n\n=== SAMPLE QUESTIONS ===')
for num_chunks in sorted(chunk_distribution.keys()):
    print(f'\n--- Questions with {num_chunks} chunk(s) ---')
    count = 0
    for q in data['questions']:
        if len(q['answer_chunk_ids']) == num_chunks:
            print(f"\nQ: {q['question']}")
            print(f"A: chunks {q['answer_chunk_ids']}")
            count += 1
            if count >= 2:
                break
