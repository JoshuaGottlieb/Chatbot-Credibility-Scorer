import numpy as np
import pandas as pd

def generate_star_total_lookup():
    star_ratings = np.arange(1,5,0.01)
    total_contribution_scores = 500 * np.arctanh((star_ratings - 1) / 4)
    df = pd.DataFrame()
    df['star_rating'] = star_ratings
    df['total_contribution'] = total_contribution_scores
#     df = pd.concat([pd.DataFrame({'star_rating': 1, 'total_contribution': 0},
#                                  index = [0]), df]).reset_index(drop = True)
    df.to_csv('contribution_totals.csv', index = False)
    
    return
    

def generate_link_contributions_lookup():
    star_ratings = np.arange(2.51,5.01,0.01)
    base_contributions = 500 * np.arctanh((star_ratings - 2.5) / 4)
    exponents = np.array([1 - ((2*(x - 1)) / 21) for x in range(1, 13)])
    full_contributions = (np.tile(base_contributions, (12, 1)) ** exponents[:, None])
    
    contribution_dict = {f'{x}_contribution':full_contributions[x - 1,:] for x in range(1,13)}
    
    df = pd.DataFrame(contribution_dict)
    df.insert(0, 'star_rating', star_ratings)
    
    df.to_csv('link_scores.csv', index = False)
    
    return

def main():
    generate_star_total_lookup()
    generate_link_contributions_lookup()
    
    return

if __name__ == '__main__':
    main()