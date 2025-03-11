from evaluation import test_sonnet

chrf_score = test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/sonnets_held_out.txt'
)

print(f"CHRF Score: {chrf_score}")