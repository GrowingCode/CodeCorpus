package com.zzk;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfWriter;

public class TextFileToPdf {

	public static void main(String[] args) {
		// 将文本文件oldTextFile.txt转换为PDF文件newPdfFile.pdf
		txtFileToPdfFile("textFile\\oldTextFile.txt", "C:\\newPdfFile.pdf");
	}

	/**
	 * 将文本文件转换为PDF文件的方法
	 * @param txtFile 原文本文件的路径
	 * @param pdfFile 生成pdf文件的路径
	 */
	private static void txtFileToPdfFile(String txtFile, String pdfFile) {
		Document doc = new Document();// 创建文档对象
		try {
			FileReader fileRead = new FileReader(txtFile);// 创建字符流对象
			BufferedReader read = new BufferedReader(fileRead);// 创建字符缓冲流对象
			PdfWriter.getInstance(doc, new FileOutputStream(pdfFile));// 关联文档和输出流对象
			doc.open();// 打开文档
			BaseFont Chinese = BaseFont.createFont("STSong-Light",
					"UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
			Font fontChinese = new Font(Chinese, 18, Font.BOLDITALIC,
					BaseColor.BLUE);// 实例化字体类、设置字体大小和颜色
			String line = null;// 存储从文本文件中读取的内容
			while ((line = read.readLine()) != null) {// 读取一行信息
				doc.add(new Paragraph(line, fontChinese));// 将读取的信息添加到文档中
			}
			doc.close();// 关闭文档对象
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
