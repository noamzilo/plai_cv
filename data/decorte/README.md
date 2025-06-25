# Padel Hit Detection Dataset

Dataset description released as part of the publication our paper: Multi-Modal Hit Detection and Positional Analysis in Padel Competitions. More information about usage of the dataset can be found in this github repository: <https://github.com/robbedec/datasets/tree/master/CVsports/Padel>.

## Dataset description

### Rallies

- The filename of the uncut tournament file, currently not public (filename)
- The onset time of the rally relative to the start of the tournament video (start)
- The offset time of the rally relative to the end of the tournament video (end)

Each rally filename is formatted using the tournament characteristics: *__DATE_LOCATION_RALLYID__*. All video's are 25 FPS and have its scoreboard hidden to improve ball tracking (the scoreboard includes a small yellow ball which confuses the model).

### Hits

Contains for each hit:

- The filename of the rally MP4 in which it occurs (filename)
- The onset time in seconds relative to the start of the rally (start)
- The onset time in seconds relative to the end of the rally (end)
- The class, which is always 'hit', but added for future extension (class)
- For class_id, see class (class_id)

### Hit assignments

- The filename of the rally name in which it occurs (video)
- Timestamp of the closest frame where the hit occurs, relative to the start of the rally (timestamp)
- Indication string of which player has hit the ball according to the assignment scheme of the paper (player) 
  - t1p1: top-left
  - t1p2: top-right
  - t2p1: bottom-left
  - t2p2: bottom right

## Citation

***Robbe Decorte, Martin Paré, Jelle Vanhaeverbeke, Joachim Taelman, Maarten Slembrouck, Steven Verstockt***; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2024, pp. 3306-3314

<https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.html>