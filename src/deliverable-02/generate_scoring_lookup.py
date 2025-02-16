import numpy as np
import pandas as pd
import os

def generate_star_total_lookup() -> None:
    """ Generates a lookup table to convert from contribution score to star rating. """
    
    # Generate star ratings from 1.00 to 4.99
    star_ratings = np.arange(1,5,0.01)
    
    # Calculate contribution scores using broadcasting and star rating function
    total_contribution_scores = 500 * np.arctanh((star_ratings - 1) / 4)
    
    # Wrap in dataframe and write to csv
    df = pd.DataFrame()
    df['star_rating'] = star_ratings
    df['total_contribution'] = total_contribution_scores
    path = './tables/contribution_totals.csv'
    if not os.path.exists(path):
        os.makedirs(os.path.join(path.split('/')[:-1]))
    df.to_csv(path, index = False)
    print(f'Contribution to star rating table created at {path}.')
    
    return

def generate_link_contributions_lookup() -> None:
    """ Generates a lookup table to convert link strength to contribution score. """
    
    # Generate star ratings from 2.51 to 5.00
    star_ratings = np.arange(2.51,5.01,0.01)
    
    # Calculate base contributions using inverse star rating function
    base_contributions = 500 * np.arctanh((star_ratings - 2.5) / 4)
    
    # Calculate contributions using decaying exponents
    exponents = np.array([1 - ((2*(x - 1)) / 21) for x in range(1, 13)])
    full_contributions = (np.tile(base_contributions, (12, 1)) ** exponents[:, None])
    
    # Wrap in dataframe and write to csv
    contribution_dict = {f'{x}_contribution':full_contributions[x - 1,:] for x in range(1,13)}
    
    df = pd.DataFrame(contribution_dict)
    df.insert(0, 'star_rating', star_ratings)
    path = './tables/link_scores.csv'
    if not os.path.exists(path):
        os.makedirs(os.path.join(path.split('/')[:-1]))
    df.to_csv(path, index = False)
    print(f'Link strength to contribution score table created at {path}.')
    
    return

def main() -> None:
    """ Main function to generate lookup tables. """
    
    generate_star_total_lookup()
    generate_link_contributions_lookup()
    
    return

if __name__ == '__main__':
    main()