-- 1. Performance Percentile Ranking
-- Purpose: Ranks players based on K/D ratio and score per minute.

WITH performance_score AS (
    SELECT
        name,
        kdRatio,
        scorePerMinute,
        wins,
        losses,
        timePlayed,
        ROUND((kdRatio * 0.5) + (scorePerMinute * 0.005), 2) AS performance_index
    FROM players
)
SELECT
    name,
    kdRatio,
    scorePerMinute,
    performance_index,
    RANK() OVER (ORDER BY performance_index DESC) AS performance_rank,
    NTILE(4) OVER (ORDER BY performance_index DESC) AS performance_quartile
FROM performance_score
ORDER BY performance_rank
LIMIT 20;

-- 2. Engagement vs Performance Matrix
--Purpose: Groups players by both engagement level and skill level.

WITH player_segments AS (
    SELECT
        name,
        kdRatio,
        scorePerMinute,
        timePlayed,
        CASE
            WHEN timePlayed < 100 THEN 'Low Engagement'
            WHEN timePlayed BETWEEN 100 AND 1000 THEN 'Medium Engagement'
            ELSE 'High Engagement'
        END AS engagement_level,
        CASE
            WHEN kdRatio < 0.5 THEN 'Struggling'
            WHEN kdRatio BETWEEN 0.5 AND 1 THEN 'Balanced'
            ELSE 'Overperforming'
        END AS skill_group
    FROM players
)
SELECT
    engagement_level,
    skill_group,
    COUNT(*) AS total_players,
    ROUND(AVG(kdRatio), 2) AS avg_kd_ratio,
    ROUND(AVG(scorePerMinute), 2) AS avg_score_per_minute,
    ROUND(AVG(timePlayed), 2) AS avg_time_played
FROM player_segments
GROUP BY engagement_level, skill_group
ORDER BY engagement_level, total_players DESC;

--3. Win Rate and Efficiency Analysis
-- Purpose: Calculates win rate and compares it with player efficiency.

WITH win_analysis AS (
    SELECT
        name,
        wins,
        losses,
        kdRatio,
        scorePerMinute,
        gamesPlayed,
        ROUND((wins * 100.0) / NULLIF(wins + losses, 0), 2) AS win_rate_percentage
    FROM players
    WHERE (wins + losses) > 0
)
SELECT
    name,
    wins,
    losses,
    win_rate_percentage,
    kdRatio,
    scorePerMinute,
    RANK() OVER (ORDER BY win_rate_percentage DESC, scorePerMinute DESC) AS win_efficiency_rank
FROM win_analysis
ORDER BY win_efficiency_rank
LIMIT 20;

--4. Accuracy-Based Skill Detection
--Purpose: Finds players with strong aiming accuracy and links it to performance.

WITH accuracy_calc AS (
    SELECT
        name,
        hits,
        shots,
        kdRatio,
        scorePerMinute,
        ROUND((hits * 100.0) / NULLIF(shots, 0), 2) AS accuracy_percentage
    FROM players
    WHERE shots > 0
)
SELECT
    name,
    accuracy_percentage,
    kdRatio,
    scorePerMinute,
    CASE
        WHEN accuracy_percentage >= 30 AND kdRatio >= 1 THEN 'High Accuracy - Strong Player'
        WHEN accuracy_percentage >= 20 AND kdRatio BETWEEN 0.5 AND 1 THEN 'Moderate Accuracy - Balanced Player'
        ELSE 'Low Accuracy - Needs Support'
    END AS accuracy_skill_label
FROM accuracy_calc
ORDER BY accuracy_percentage DESC
LIMIT 20;

--5. Player Drop-Off Risk Detection
--Purpose: Detects players who may stop playing because of weak performance or low engagement.

WITH risk_features AS (
    SELECT
        name,
        kdRatio,
        scorePerMinute,
        timePlayed,
        gamesPlayed,
        wins,
        losses,
        CASE
            WHEN timePlayed < 100 AND kdRatio < 0.5 THEN 'High Drop-Off Risk'
            WHEN timePlayed < 500 AND kdRatio BETWEEN 0.5 AND 1 THEN 'Medium Drop-Off Risk'
            ELSE 'Low Drop-Off Risk'
        END AS dropoff_risk
    FROM players
)
SELECT
    dropoff_risk,
    COUNT(*) AS total_players,
    ROUND(AVG(kdRatio), 2) AS avg_kd_ratio,
    ROUND(AVG(scorePerMinute), 2) AS avg_score_per_minute,
    ROUND(AVG(timePlayed), 2) AS avg_time_played,
    ROUND(AVG(gamesPlayed), 2) AS avg_games_played
FROM risk_features
GROUP BY dropoff_risk
ORDER BY total_players DESC;

--6. Difficulty Recommendation Validation
-- Purpose: Checks whether recommendations match player performance.

WITH joined_data AS (
    SELECT
        p.name,
        p.kdRatio,
        p.scorePerMinute,
        p.timePlayed,
        r.Final_Group,
        r.Difficulty_Recommendation
    FROM players p
    JOIN recommendations r
        ON p.name = r.Player_ID
)
SELECT
    Final_Group,
    Difficulty_Recommendation,
    COUNT(*) AS total_players,
    ROUND(AVG(kdRatio), 2) AS avg_kd_ratio,
    ROUND(AVG(scorePerMinute), 2) AS avg_score_per_minute,
    ROUND(AVG(timePlayed), 2) AS avg_time_played
FROM joined_data
GROUP BY Final_Group, Difficulty_Recommendation
ORDER BY total_players DESC;

-- 7. Outlier Detection in Player Performance
-- Purpose: Finds unusual players with extremely high or low performance.

WITH stats AS (
    SELECT
        AVG(scorePerMinute) AS avg_spm,
        AVG(kdRatio) AS avg_kd
    FROM players
),
outlier_check AS (
    SELECT
        p.name,
        p.kdRatio,
        p.scorePerMinute,
        p.timePlayed,
        s.avg_spm,
        s.avg_kd,
        CASE
            WHEN p.scorePerMinute > s.avg_spm * 2 THEN 'High Performance Outlier'
            WHEN p.scorePerMinute < s.avg_spm * 0.25 THEN 'Low Performance Outlier'
            ELSE 'Normal'
        END AS outlier_type
    FROM players p
    CROSS JOIN stats s
)
SELECT
    name,
    kdRatio,
    scorePerMinute,
    timePlayed,
    outlier_type
FROM outlier_check
WHERE outlier_type != 'Normal'
ORDER BY scorePerMinute DESC;

--8. Advanced Player Scoring Index
-- Purpose: Creates a custom weighted score for ranking players

WITH scoring AS (
    SELECT
        name,
        kdRatio,
        scorePerMinute,
        wins,
        losses,
        timePlayed,
        gamesPlayed,
        ROUND(
            (kdRatio * 40) +
            (scorePerMinute * 0.30) +
            (wins * 0.20) +
            (timePlayed * 0.01),
            2
        ) AS player_score_index
    FROM players
)
SELECT
    name,
    kdRatio,
    scorePerMinute,
    wins,
    losses,
    timePlayed,
    player_score_index,
    CASE
        WHEN player_score_index >= 150 THEN 'Elite'
        WHEN player_score_index >= 80 THEN 'Competitive'
        WHEN player_score_index >= 40 THEN 'Casual'
        ELSE 'Beginner'
    END AS player_level
FROM scoring
ORDER BY player_score_index DESC
LIMIT 20;

