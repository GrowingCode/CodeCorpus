package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;


public class SetParagraphLeftRightIndent {
	
	
	public static void main(String[] args){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("C:\\Java编程词典.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			Paragraph paragraph1 = new Paragraph("The Paragraph IndentLeft");// 实例化段落并添加内容
			paragraph1.setIndentationLeft(100);// 段落左缩进
			document.add(paragraph1);// 向文档中添加段落
			Paragraph paragraph2 = new Paragraph("The Paragraph IndentRight");// 实例化段落并添加内容
			paragraph2.setIndentationRight(100);// 段落右缩进
			document.add(paragraph2);// 向文档中添加段落
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}