import time
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from groundino_sam21_cl import GroundedSAM2Client  # 导入客户端类


class PerformanceTester:
    def __init__(self, client, image_dir, num_runs=10):
        self.client = client
        self.image_dir = Path(image_dir)
        self.num_runs = num_runs
        self.image_paths = self._get_image_paths()
        self.test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = Path(f"./grounded_sam2_perf_results_{self.test_timestamp}")
        self.result_dir.mkdir(exist_ok=True)

        # 验证图片数量
        if len(self.image_paths) < num_runs:
            raise ValueError(f"需要至少{num_runs}张测试图片，仅找到{len(self.image_paths)}张")

    def _get_image_paths(self):
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        paths = []
        for file in os.listdir(self.image_dir):
            if any(file.lower().endswith(ext) for ext in img_extensions):
                full_path = self.image_dir / file
                paths.append(str(full_path))
        return paths[:self.num_runs]  # 取前N张图片（N=num_runs）

    def _save_json(self, data, filename: str):
        file_path = self.result_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"📄 数据已保存到: {file_path}")
        return file_path

    def test_grounding_dino(self, text_prompt="person. car. dog. Burgers. Pizza. bicycles. boat. mouses. keyboards. tv. microwave. oven. refrigerator. bottles. banana. oranges. chairs. potted plant."):
        print("="*60)
        print("开始测试 Grounding DINO 性能...")
        print("="*60)

        times = []
        detailed_results = []

        for i, img_path in enumerate(self.image_paths, 1):
            img_filename = Path(img_path).name
            start_time = time.perf_counter()
            try:
                det_result = self.client.detect_objects(img_path, text_prompt)
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)

                # 记录单张图片的详细信息
                detailed_results.append({
                    "test_index": i,
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "text_prompt": text_prompt,
                    "inference_time_s": round(elapsed, 4),
                    "bboxes_xyxy": det_result["boxes"],
                    "confidences": det_result["confidences"],
                    "labels": det_result["labels"],
                    "image_height": det_result["image_height"],
                    "image_width": det_result["image_width"],
                    "success": True
                })

                print(f"✅ 完成第{i}/{self.num_runs}张 [{img_filename}]，耗时: {elapsed:.4f}秒")

            except Exception as e:
                detailed_results.append({
                    "test_index": i,
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "error": str(e),
                    "success": False
                })
                print(f"❌ 第{i}张 [{img_filename}] 测试失败: {str(e)}")

        # 保存详细结果
        self._save_json(detailed_results, f"grounding_dino_detailed_results_{self.test_timestamp}.json")

        # 计算性能指标
        valid_times = [r["inference_time_s"] for r in detailed_results if r["success"]]
        if not valid_times:
            print("\n⚠️  无有效测试结果，跳过性能指标计算")
            return None

        avg_time = np.mean(valid_times)
        std_time = np.std(valid_times) if len(valid_times) > 1 else 0.0
        min_time = np.min(valid_times)
        max_time = np.max(valid_times)
        total_valid = len(valid_times)
        throughput = total_valid / np.sum(valid_times)

        perf_results = {
            "模型": "Grounding DINO",
            "测试时间": datetime.now().isoformat(),
            "测试图片总数": self.num_runs,
            "成功测试数": total_valid,
            "失败测试数": self.num_runs - total_valid,
            "平均耗时(秒)": round(avg_time, 4),
            "耗时标准差(秒)": round(std_time, 4),
            "最小耗时(秒)": round(min_time, 4),
            "最大耗时(秒)": round(max_time, 4),
            "吞吐量(张/秒)": round(throughput, 4),
            "测试提示词": text_prompt,
            "详细结果文件路径": f"grounding_dino_detailed_results_{self.test_timestamp}.json"
        }

        self._save_json(perf_results, f"grounding_dino_perf_summary_{self.test_timestamp}.json")
        return perf_results

    def test_sam2(self, text_prompt="person. car. dog. Burgers. Pizza. bicycles. boat. mouses. keyboards. tv. microwave. oven. refrigerator. bottles. banana. oranges. chairs. potted plant."):
        print("\n" + "="*60)
        print("开始测试 SAM2 性能...")
        print("="*60)

        times = []
        detailed_results = []
        all_boxes = []

        print("\n🔍 预获取所有图片的检测框（Grounding DINO）...")
        for img_path in self.image_paths:
            try:
                det_result = self.client.detect_objects(img_path, text_prompt)
                all_boxes.append({
                    "image_path": img_path,
                    "boxes": det_result["boxes"],
                    "image_height": det_result["image_height"],
                    "image_width": det_result["image_width"]
                })
                print(f"   ✅ 已获取 [{Path(img_path).name}] 的检测框（共{len(det_result['boxes'])}个目标）")
            except Exception as e:
                all_boxes.append({"error": str(e), "image_path": img_path})
                print(f"   ❌ 获取 [{Path(img_path).name}] 检测框失败: {e}")
                return None

        print("\n🚀 开始SAM2分割测试...")
        for i, (img_info, img_path) in enumerate(zip(all_boxes, self.image_paths), 1):
            img_filename = Path(img_path).name
            boxes = img_info["boxes"]
            start_time = time.perf_counter()

            if not boxes:
                detailed_results.append({
                    "test_index": i,
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "error": "无检测框（Grounding DINO未检测到目标）",
                    "success": False
                })
                print(f"❌ 第{i}张 [{img_filename}] 无检测框，跳过测试")
                continue

            try:
                seg_result = self.client.segment_objects(img_path, boxes)
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)

                detailed_results.append({
                    "test_index": i,
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "inference_time_s": round(elapsed, 4),
                    "input_bboxes_xyxy": boxes,
                    "segmentation_masks_rle": seg_result["masks_rle"],
                    "segmentation_scores": seg_result["scores"],
                    "image_height": seg_result["image_height"],
                    "image_width": seg_result["image_width"],
                    "success": True
                })

                print(f"✅ 完成第{i}/{self.num_runs}张 [{img_filename}]，耗时: {elapsed:.4f}秒（{len(boxes)}个目标）")

            except Exception as e:
                detailed_results.append({
                    "test_index": i,
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "input_bboxes_xyxy": boxes,
                    "error": str(e),
                    "success": False
                })
                print(f"❌ 第{i}张 [{img_filename}] 测试失败: {str(e)}")

        # 保存详细结果
        self._save_json(detailed_results, f"sam2_detailed_results_{self.test_timestamp}.json")

        # 计算性能指标
        valid_times = [r["inference_time_s"] for r in detailed_results if r["success"]]
        if not valid_times:
            print("\n⚠️  无有效测试结果，跳过性能指标计算")
            return None

        avg_time = np.mean(valid_times)
        std_time = np.std(valid_times) if len(valid_times) > 1 else 0.0
        min_time = np.min(valid_times)
        max_time = np.max(valid_times)
        total_valid = len(valid_times)
        throughput = total_valid / np.sum(valid_times)

        perf_results = {
            "模型": "SAM2（基于Grounding DINO BBox）",
            "测试时间": datetime.now().isoformat(),
            "测试图片总数": self.num_runs,
            "成功测试数": total_valid,
            "失败测试数": self.num_runs - total_valid,
            "平均耗时(秒)": round(avg_time, 4),
            "耗时标准差(秒)": round(std_time, 4),
            "最小耗时(秒)": round(min_time, 4),
            "最大耗时(秒)": round(max_time, 4),
            "吞吐量(张/秒)": round(throughput, 4),
            "输入BBox来源": "Grounding DINO检测结果",
            "详细结果文件路径": f"sam2_detailed_results_{self.test_timestamp}.json"
        }

        self._save_json(perf_results, f"sam2_perf_summary_{self.test_timestamp}.json")
        return perf_results

    def test_combined(self, text_prompt="person. car. dog. Burgers. Pizza. bicycles. boat. mouses. keyboards. tv. microwave. oven. refrigerator. bottles. banana. oranges. chairs. potted plant."):
        """测试端到端（Grounding DINO + SAM2）性能"""
        print("\n" + "="*60)
        print("开始测试 端到端（Grounding DINO + SAM2）性能...")
        print("="*60)

        times = []
        detailed_results = []

        for i, img_path in enumerate(self.image_paths, 1):
            img_filename = Path(img_path).name
            start_time = time.perf_counter()
            try:
                combined_result = self.client.detect_and_segment(img_path, text_prompt)
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)

                detailed_results.append({
                    "test_index": i,
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "text_prompt": text_prompt,
                    "end2end_inference_time_s": round(elapsed, 4),
                    "grounding_dino_bboxes_xyxy": combined_result["boxes"],
                    "grounding_dino_confidences": combined_result["confidences"],
                    "grounding_dino_labels": combined_result["labels"],
                    "sam2_masks_rle": combined_result["masks_rle"],
                    "sam2_segmentation_scores": combined_result["segmentation_scores"],
                    "image_height": combined_result["image_height"],
                    "image_width": combined_result["image_width"],
                    "success": True
                })

                print(f"✅ 完成第{i}/{self.num_runs}张 [{img_filename}]，耗时: {elapsed:.4f}秒")

            except Exception as e:
                detailed_results.append({
                    "test_index": i,
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "error": str(e),
                    "success": False
                })
                print(f"❌ 第{i}张 [{img_filename}] 测试失败: {str(e)}")

        # 保存详细结果
        self._save_json(detailed_results, f"combined_end2end_detailed_results_{self.test_timestamp}.json")

        # 计算性能指标
        valid_times = [r["end2end_inference_time_s"] for r in detailed_results if r["success"]]
        if not valid_times:
            print("\n⚠️  无有效测试结果，跳过性能指标计算")
            return None

        avg_time = np.mean(valid_times)
        std_time = np.std(valid_times) if len(valid_times) > 1 else 0.0
        min_time = np.min(valid_times)
        max_time = np.max(valid_times)
        total_valid = len(valid_times)
        throughput = total_valid / np.sum(valid_times)

        perf_results = {
            "模型": "Grounding DINO + SAM2（端到端）",
            "测试时间": datetime.now().isoformat(),
            "测试图片总数": self.num_runs,
            "成功测试数": total_valid,
            "失败测试数": self.num_runs - total_valid,
            "平均端到端耗时(秒)": round(avg_time, 4),
            "耗时标准差(秒)": round(std_time, 4),
            "最小耗时(秒)": round(min_time, 4),
            "最大耗时(秒)": round(max_time, 4),
            "吞吐量(张/秒)": round(throughput, 4),
            "测试提示词": text_prompt,
            "详细结果文件路径": f"combined_end2end_detailed_results_{self.test_timestamp}.json"
        }

        self._save_json(perf_results, f"combined_end2end_perf_summary_{self.test_timestamp}.json")
        return perf_results

    def print_results(self, results):
        if not results:
            print("\n⚠️  无有效测试结果可打印")
            return

        print("\n" + "="*60)
        print(f"{results['模型']} 性能测试总结")
        print("="*60)
        for key, value in results.items():
            if key not in ["详细结果文件路径", "测试提示词"]:
                print(f"{key:20s}: {value}")
        print(f"详细结果文件路径: {results['详细结果文件路径']}")
        print("="*60 + "\n")

    def generate_final_report(self, dino_results, sam_results, combined_results):
        """生成最终汇总报告（修复键名不一致问题）"""
        # 统一SAM2结果的键名格式
        if sam_results and sam_results != "测试失败":
            # 将SAM2的"平均耗时(秒)"重命名为与其他模型一致的键
            sam_results["平均耗时(秒)"] = sam_results.get("平均耗时(秒)", 0)

        # 统一端到端模型的键名
        if combined_results and combined_results != "测试失败":
            # 复制端到端耗时到通用键名
            combined_results["平均耗时(秒)"] = combined_results.get("平均端到端耗时(秒)", 0)

        final_report = {
            "测试汇总信息": {
                "测试开始时间": datetime.strptime(self.test_timestamp, "%Y%m%d_%H%M%S").isoformat(),
                "测试图片总数": self.num_runs,
                "测试图片目录": str(self.image_dir),
                "测试提示词": self.test_grounding_dino.__defaults__[0],  # 获取默认提示词
                "结果保存目录": str(self.result_dir),
                "测试设备API地址": self.client.base_url
            },
            "各模型核心性能指标对比": {
                "Grounding DINO": dino_results if dino_results else "测试失败",
                "SAM2（基于DINO BBox）": sam_results if sam_results else "测试失败",
                "端到端（DINO+SAM2）": combined_results if combined_results else "测试失败"
            },
            "指标说明": {
                "平均耗时": "单张图片的平均推理时间（秒），越小性能越好",
                "吞吐量": "每秒可处理的图片数量（张/秒），越大性能越好",
                "耗时标准差": "推理时间的波动程度，越小稳定性越好",
                "BBoxes_xyxy": "目标检测框坐标（x1,y1,x2,y2格式，可直接用于后续任务）",
                "masks_rle": "分割掩码（RLE压缩格式，需用pycocotools解码）"
            }
        }

        # 保存最终汇总报告
        report_path = self._save_json(final_report, f"grounded_sam2_final_report_{self.test_timestamp}.json")

        # 生成文本格式报告（便于快速阅读）
        text_report = f"""
# Grounded-SAM2 性能测试最终报告
{'='*80}
## 测试基本信息
- 测试时间: {final_report['测试汇总信息']['测试开始时间']}
- 测试图片数量: {final_report['测试汇总信息']['测试图片总数']}
- 图片目录: {final_report['测试汇总信息']['测试图片目录']}
- API地址: {final_report['测试汇总信息']['测试设备API地址']}
- 结果保存目录: {final_report['测试汇总信息']['结果保存目录']}

## 核心性能指标对比
{'='*80}
"""

        # 遍历各模型，添加性能对比
        for model_name, model_data in final_report["各模型核心性能指标对比"].items():
            if model_data == "测试失败":
                text_report += f"\n### {model_name}\n- 状态: 测试失败\n"
                continue

            text_report += f"""
### {model_name}
- 成功测试数: {model_data['成功测试数']}/{model_data['测试图片总数']}
- 平均耗时: {model_data['平均耗时(秒)']} 秒
- 耗时标准差: {model_data['耗时标准差(秒)']} 秒
- 最小耗时: {model_data['最小耗时(秒)']} 秒
- 最大耗时: {model_data['最大耗时(秒)']} 秒
- 吞吐量: {model_data['吞吐量(张/秒)']} 张/秒
- 详细结果文件: {model_data['详细结果文件路径']}
"""

        # 添加指标说明
        text_report += f"""
## 指标说明
{'='*80}
"""
        for indicator, desc in final_report["指标说明"].items():
            text_report += f"- {indicator}: {desc}\n"

        # 保存文本报告
        text_report_path = self.result_dir / f"grounded_sam2_final_report_{self.test_timestamp}.txt"
        with open(text_report_path, "w", encoding="utf-8") as f:
            f.write(text_report)
        print(f"\n📄 最终汇总报告已保存到: {text_report_path}")

        return final_report


if __name__ == "__main__":
    # ===================== 配置测试参数 =====================
    IMAGE_DIR = "./img_src"  # 测试图片目录（需提前放入至少10张图片）
    NUM_RUNS = 10            # 测试图片数量（固定10张）
    # 检测提示词（可根据需求调整，需以句号结尾）
    TEXT_PROMPT = "person. car. dog. Burgers. Pizza. bicycles. boat. mouses. keyboards. tv. microwave. oven. refrigerator. bottles. banana. oranges. chairs. potted plant."

    # ===================== 初始化与执行测试 =====================
    try:
        # 初始化客户端（连接API服务）
        client = GroundedSAM2Client(base_url="http://localhost:5000")
        print(f"✅ 已连接到API服务: {client.base_url}")

        # 初始化测试器
        tester = PerformanceTester(client, IMAGE_DIR, NUM_RUNS)
        print(f"✅ 测试器初始化完成，待测试图片: {len(tester.image_paths)} 张")

        # 执行三个核心测试
        dino_perf = tester.test_grounding_dino(TEXT_PROMPT)
        tester.print_results(dino_perf)

        sam_perf = tester.test_sam2(TEXT_PROMPT)
        tester.print_results(sam_perf)

        combined_perf = tester.test_combined(TEXT_PROMPT)
        tester.print_results(combined_perf)

        # 生成最终汇总报告
        print("\n" + "="*80)
        print("开始生成最终汇总报告...")
        tester.generate_final_report(dino_perf, sam_perf, combined_perf)
        print("="*80)
        print("🎉 所有测试完成！结果已保存到: ", tester.result_dir)

    except Exception as e:
        print(f"\n❌ 测试初始化失败: {str(e)}")
        exit(1)