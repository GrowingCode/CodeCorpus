package com.zzk;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfReader;
import com.itextpdf.text.pdf.PdfStamper;
import com.itextpdf.text.pdf.PdfWriter;

public class EditPdfDocumentDemo {
	public static void main(String[] args) {
		createOldFile();// 创建原文件
		editOldFile();// 编辑原文件
	}

	public static void createOldFile() {// 创建原文件的方法
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter
					.getInstance(document, new FileOutputStream("c:\\原文档.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			document.add(new Paragraph("First")); // 向文档中添加内容
			document.newPage();
			document.add(new Paragraph("Second")); // 向文档中添加内容
			document.newPage();
			document.add(new Paragraph("Third")); // 向文档中添加内容
			document.close();// 关闭文档对象
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}

	public static void editOldFile() {// 编辑原文件的方法
		try {
			PdfReader reader = new PdfReader("c:\\原文档.pdf");// 创建“原文档.pdf”的PdfReader对象
			int totalPages = reader.getNumberOfPages();// 获得总页数
			PdfStamper stamp = new PdfStamper(reader, new FileOutputStream(
					"c:\\编辑后文档的临时文件.pdf"));// 创建PdfStamper对象
			BaseFont chinese = BaseFont.createFont("STSong-Light",
					"UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义字体
			PdfContentByte under = null;
			for (int i = 1; i <= totalPages; i++) {
				under = stamp.getUnderContent(i);// 获得每一页的内容
				under.beginText();// 标记文本开始
				under.setFontAndSize(chinese, 18);// 设置字体和字号
				under.setTextMatrix(200, 810);// 设置页码的显示位置
				under.showText("第" + i + "页");// 添加页脚
				under.endText();// 标记文本结束
				under.beginText();// 标记文本开始
				under.setFontAndSize(chinese, 32);// 设置字体和字号
				under.setTextMatrix(100, 750);// 设置文本的显示位置
				under.showText("新添加的内容" + i);// 添加新文本
				under.endText();// 标记文本结束
			}
			stamp.close();// PdfStamper对象，将从“原文档.pdf”中读取的文档添加页码后写入“编辑后文档的临时文件.pdf”
			File oldFile = new File("c:\\原文档.pdf");// 创建原文件的File对象
			oldFile.delete();// 删除原文件
			File tempFile = new File("c:\\编辑后文档的临时文件.pdf");// 创建临时文件的File对象
			tempFile.renameTo(oldFile);// 重命名临时文件为原文件名
			tempFile.delete();// 删除临时文件
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
