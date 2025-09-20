from pathlib import Path
from sftk.utils import read_file_to_df
import pandas as pd
import time


def process_biigle_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Processes raw Biigle annotations to count species per frame and format data."""
    # 1. Select required columns
    required_columns = ["label_name", "video_filename", "frames"]
    projected_df = df[required_columns]

    # 2. Group and count species occurrences
    grouped_df = projected_df.groupby(["video_filename", "frames", "label_name"]).size().reset_index(name='count')

    # 3. Calculate TimeOfMax
    # Extract start seconds from filename (e.g., ..._clip_115_30... -> 115)
    grouped_df['start_seconds'] = grouped_df['video_filename'].str.split('_clip_').str[1].str.split('_').str[0].astype(
        int)
    # Extract frame seconds from the '[seconds]' string
    grouped_df['frame_seconds'] = grouped_df['frames'].str.strip('[]').astype(float)
    # Calculate total seconds and format to HH:MM:SS
    total_seconds = grouped_df['start_seconds'] + grouped_df['frame_seconds']
    grouped_df['TimeOfMax'] = total_seconds.apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)))

    # 4. Rename columns to match target format
    renamed_df = grouped_df.rename(columns={
        'video_filename': 'DropID',
        'label_name': 'ScientificName',
        'count': 'MaxInterval'
    })

    # Trim DropID to exclude .mp4 and subsequent parts
    renamed_df['DropID'] = renamed_df['DropID'].str.split('.mp4').str[0]

    # 5. Add constant columns (may change according to video)
    renamed_df['AnnotatedBy'] = 'expert'
    renamed_df['IntervalAnnotation'] = 30
    renamed_df['ConfidenceAgreement'] = 'NA'

    # 6. Select and reorder final columns
    final_df = renamed_df[
        ['DropID', 'ScientificName', 'TimeOfMax', 'MaxInterval', 'AnnotatedBy', 'IntervalAnnotation',
         'ConfidenceAgreement']]

    return final_df


def main():
    """Main function to read, process, and save Biigle annotation data."""
    repo_root = Path(__file__).resolve().parents[1]
    biigle_file_path = repo_root / "data/25516-ton-20221205-buv-ton-044-01.csv"

    biigle_df = read_file_to_df(str(biigle_file_path))

    if not biigle_df.empty:
        processed_df = process_biigle_annotations(biigle_df)

        # Define output path
        output_filename = f"{biigle_file_path.stem}_parsed.csv"
        output_path = biigle_file_path.with_name(output_filename)

        # Save the processed dataframe to a new csv file
        processed_df.to_csv(output_path, index=False)
        print(f"Processed file saved successfully to: {output_path}")
    else:
        print(f"DataFrame is empty. Please check the file path: {biigle_file_path}")


if __name__ == "__main__":
    main()
