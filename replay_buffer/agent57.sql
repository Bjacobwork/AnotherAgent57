SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";
CREATE DATABASE IF NOT EXISTS `agent57` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `agent57`;
CREATE TABLE IF NOT EXISTS `episode` (`episode_id` int(11) NOT NULL, `policy` int(11) NOT NULL, `priority` float DEFAULT NULL, PRIMARY KEY (`episode_id`)) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE TABLE IF NOT EXISTS `trace` ( `episode_id` int(11) NOT NULL, `initial_step` int(11) NOT NULL, `trace_id` int(11) NOT NULL AUTO_INCREMENT, `priority` float NOT NULL, PRIMARY KEY (`episode_id`,`initial_step`), UNIQUE KEY `trace_id` (`trace_id`)) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE TABLE IF NOT EXISTS `transition` (`episode_id` int(11) NOT NULL, `step` int(11) NOT NULL,  `prev_extrinsic_reward` float NOT NULL,  `prev_intrinsic_reward` float NOT NULL,  `action` int(11) NOT NULL,  `observation` blob NOT NULL,  `hidden_state` blob NOT NULL,  `mu` float NOT NULL,  `q_value` float NOT NULL, `discounted_q` float NOT NULL,  PRIMARY KEY (`episode_id`,`step`)) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;